#include "system/BundleAdjusterCeres.h"
#include "PreKeyFrameEntryInternals.h"
#include "system/AffineLightTransform.h"
#include "system/SphericalPlus.h"
#include "util/defs.h"
#include "util/geometry.h"
#include "util/util.h"
#include <ceres/ceres.h>
#include <ceres/local_parameterization.h>

#define PS (settings.residualPattern.pattern().size())
#define PH (settings.residualPattern.height)

namespace mdso {

BundleAdjusterCeres::~BundleAdjusterCeres() {}

struct DirectResidual {
  DirectResidual(PreKeyFrameEntryInternals::Interpolator_t *hostFrame,
                 PreKeyFrameEntryInternals::Interpolator_t *targetFrame,
                 const CameraModel *targetCam, OptimizedPoint *optimizedPoint,
                 const Vec2 &pos, const SE3 &hostFrameToBody,
                 const SE3 &targetBodyToFrame, KeyFrame *hostKf,
                 KeyFrame *targetKf, const Settings::Depth &depthSettings)
      : targetCam(targetCam)
      , hostDirection(targetCam->unmap(pos).normalized())
      , hostFrameToBody(hostFrameToBody)
      , targetBodyToFrame(targetBodyToFrame)
      , targetFrame(targetFrame)
      , optimizedPoint(optimizedPoint)
      , hostKf(hostKf)
      , targetKf(targetKf)
      , depthSettings(depthSettings) {
    hostFrame->Evaluate(pos[1], pos[0], &hostIntensity);
  }

  template <typename T>
  bool operator()(const T *const depthParamP, const T *const hostTransP,
                  const T *const hostRotP, const T *const targetTransP,
                  const T *const targetRotP, const T *const hostAffP,
                  const T *const targetAffP, T *res) const {
    using Vec2t = Eigen::Matrix<T, 2, 1>;
    using Vec3t = Eigen::Matrix<T, 3, 1>;
    using Mat33t = Eigen::Matrix<T, 3, 3>;
    using Quatt = Eigen::Quaternion<T>;
    using SE3t = Sophus::SE3<T>;

    Eigen::Map<const Vec3t> hostTransM(hostTransP);
    Vec3t hostTrans(hostTransM);
    Eigen::Map<const Quatt> hostRotM(hostRotP);
    Quatt hostRot(hostRotM);
    SE3t hostToWorld(hostRot, hostTrans);

    Eigen::Map<const Vec3t> targetTransM(targetTransP);
    Vec3t targetTrans(targetTransM);
    Eigen::Map<const Quatt> targetRotM(targetRotP);
    Quatt targetRot(targetRotM);
    SE3t targetToWorld(targetRot, targetTrans);

    SE3t targetBodyToFrameT = targetBodyToFrame.template cast<T>();
    SE3t hostFrameToBodyT = hostFrameToBody.template cast<T>();

    const T *hostAffLightP = hostAffP;
    AffineLightTransform<T> lightWorldToHost(hostAffLightP[0],
                                             hostAffLightP[1]);

    const T *targetAffLightP = targetAffP;
    AffineLightTransform<T> lightWorldToTarget(targetAffLightP[0],
                                               targetAffLightP[1]);

    T depth = depthSettings.useMinPlusExpParametrization
                  ? depthSettings.min + ceres::exp(*depthParamP)
                  : ceres::exp(*depthParamP);

    Vec3t targetPos;
    if (depth > T(depthSettings.max))
      targetPos = targetBodyToFrameT.so3() *
                  (targetToWorld.so3().inverse() *
                   (hostToWorld.so3() *
                    (hostFrameToBodyT.so3() * (hostDirection.cast<T>()))));
    else
      targetPos = targetBodyToFrameT *
                  (targetToWorld.inverse() *
                   (hostToWorld *
                    (hostFrameToBodyT * (hostDirection.cast<T>() * depth))));
    Vec2t targetPosMapped = targetCam->map(targetPos);
    T trackedIntensity;
    targetFrame->Evaluate(targetPosMapped[1], targetPosMapped[0],
                          &trackedIntensity);
    T transformedHostIntensity =
        lightWorldToTarget(lightWorldToHost.inverse()(T(hostIntensity)));
    *res = trackedIntensity - transformedHostIntensity;

    return true;
  }

  const CameraModel *targetCam;
  Vec3 hostDirection;
  double hostIntensity;
  SE3 hostFrameToBody, targetBodyToFrame;
  PreKeyFrameEntryInternals::Interpolator_t *targetFrame;
  OptimizedPoint *optimizedPoint;
  KeyFrame *hostKf;
  KeyFrame *targetKf;
  Settings::Depth depthSettings;
};

void logOobStatistics(const std::vector<DirectResidual *> &residuals,
                      const BundleAdjusterSettings &settings) {
  int numOob = 0, numOobBigDepth = 0;
  for (const DirectResidual *res : residuals) {
    SE3 hostToWorld = res->hostKf->thisToWorld();
    SE3 hostCamToBody = res->hostFrameToBody;
    SE3 targetToWorld = res->targetKf->thisToWorld();
    SE3 targetBodyToCam = res->targetBodyToFrame;
    SE3 hostToTarget =
        targetBodyToCam * targetToWorld.inverse() * hostToWorld * hostCamToBody;
    double depth = res->optimizedPoint->depth();
    Vec2 reproj =
        res->targetCam->map(hostToTarget * (depth * res->optimizedPoint->dir));

    if (!res->targetCam->isOnImage(reproj, PH)) {
      numOob++;
      if (depth > settings.depth.max)
        numOobBigDepth++;
    }
  }
  LOG(INFO) << "total resduals; total OOB; OOB with big depths:";
  LOG(INFO) << residuals.size() << ' ' << numOob << ' ' << numOobBigDepth;
}

void BundleAdjusterCeres::adjust(KeyFrame **keyFrames, int numKeyFrames,
                                 const BundleAdjusterSettings &settings) const {
  CHECK_GE(numKeyFrames, 2);

  CameraBundle *cam = keyFrames[0]->preKeyFrame->cam;
  StdVector<SE3> bodyToWorld;
  std::vector<std::pair<OptimizedPoint *, double>> depthParams;
  depthParams.reserve(Settings::max_maxOptimizedPoints);
  auto oldDepthParamsBegin = depthParams.begin();

  int pointsTotal = 0, pointsOOB = 0;

  std::shared_ptr<ceres::ParameterBlockOrdering> ordering(
      new ceres::ParameterBlockOrdering());

  ceres::Problem problem;

  bodyToWorld.reserve(numKeyFrames);
  for (int i = 0; i < numKeyFrames; ++i) {
    KeyFrame *keyFrame = keyFrames[i];
    bodyToWorld.push_back(keyFrame->thisToWorld());

    problem.AddParameterBlock(bodyToWorld.back().translation().data(), 3);
    problem.AddParameterBlock(bodyToWorld.back().so3().data(), 4,
                              new ceres::EigenQuaternionParameterization());

    ordering->AddElementToGroup(bodyToWorld.back().translation().data(), 1);
    ordering->AddElementToGroup(bodyToWorld.back().so3().data(), 1);

    for (KeyFrameEntry &entry : keyFrame->frames) {
      double *affLight = entry.lightWorldToThis.data;
      problem.AddParameterBlock(affLight, 2);
      problem.SetParameterLowerBound(affLight, 0,
                                     settings.affineLight.minAffineLightA);
      problem.SetParameterUpperBound(affLight, 0,
                                     settings.affineLight.maxAffineLightA);
      problem.SetParameterLowerBound(affLight, 1,
                                     settings.affineLight.minAffineLightB);
      problem.SetParameterUpperBound(affLight, 1,
                                     settings.affineLight.maxAffineLightB);
      if (!settings.affineLight.optimizeAffineLight)
        problem.SetParameterBlockConstant(affLight);
      ordering->AddElementToGroup(affLight, 1);
    }
  }

  problem.SetParameterBlockConstant(bodyToWorld[0].translation().data());
  problem.SetParameterBlockConstant(bodyToWorld[0].so3().data());
  for (KeyFrameEntry &entry : keyFrames[0]->frames)
    problem.SetParameterBlockConstant(entry.lightWorldToThis.data);

  SE3 firstToWorld = bodyToWorld[0];
  SE3 secondToWorld = bodyToWorld[1];
  double radius =
      (secondToWorld.translation() - firstToWorld.translation()).norm();
  Vec3 center = firstToWorld.translation();
  if (radius > settings.optimization.minFirstToSecondRadius)
    problem.SetParameterization(
        bodyToWorld[1].translation().data(),
        new ceres::AutoDiffLocalParameterization<SphericalPlus, 3, 2>(
            new SphericalPlus(center, radius, secondToWorld.translation())));
  else {
    problem.SetParameterBlockConstant(bodyToWorld[1].translation().data());
    LOG(WARNING) << "FirstToSecond distance too small, spherical "
                    "parametrization was not applied";
  }

  if (settings.optimization.fixedRotationOnSecondKF)
    problem.SetParameterBlockConstant(bodyToWorld[1].so3().data());

  if (settings.optimization.fixedMotionOnFirstAdjustent && numKeyFrames == 2) {
    problem.SetParameterBlockConstant(bodyToWorld[1].translation().data());
    problem.SetParameterBlockConstant(bodyToWorld[1].so3().data());
  }

  std::vector<DirectResidual *> residuals;

  int numPoints = 0;
  int numBadDepths = 0, numOobSmallDepths = 0, numOobLargeDepths = 0;
  for (int hostInd = 0; hostInd < numKeyFrames; ++hostInd) {
    KeyFrame *hostFrame = keyFrames[hostInd];
    for (int hostCamInd = 0; hostCamInd < cam->bundle.size(); ++hostCamInd) {
      KeyFrameEntry &hostEntry = hostFrame->frames[hostCamInd];
      for (auto &op : hostEntry.optimizedPoints) {
        numPoints++;
        if ((settings.depth.setMinBound ||
             settings.depth.useMinPlusExpParametrization) &&
            op.depth() <= settings.depth.min) {
          numOobSmallDepths++;
          continue;
        }
        if (settings.depth.setMaxBound && op.depth() > settings.depth.max) {
          numOobLargeDepths++;
          continue;
        }
        if (op.depth() < 0 || !std::isfinite(op.depth())) {
          numBadDepths++;
          continue;
        }

        depthParams.emplace_back(&op, std::log(op.depth()));
        double *depthParamPtr = &depthParams.back().second;
        problem.AddParameterBlock(depthParamPtr, 1);
        if (settings.depth.useMinPlusExpParametrization)
          *depthParamPtr = std::log(op.depth() - settings.depth.min);
        else if (settings.depth.setMinBound)
          problem.SetParameterLowerBound(depthParamPtr, 0,
                                         std::log(settings.depth.min));
        if (settings.depth.setMaxBound) {
          if (settings.depth.useMinPlusExpParametrization)
            problem.SetParameterUpperBound(
                depthParamPtr, 0,
                std::log(settings.depth.max - settings.depth.min));
          else
            problem.SetParameterUpperBound(depthParamPtr, 0,
                                           std::log(settings.depth.max));
        }

        ordering->AddElementToGroup(depthParamPtr, 0);

        for (int targetInd = 0; targetInd < numKeyFrames; ++targetInd) {
          KeyFrame *targetFrame = keyFrames[targetInd];
          for (int targetCamInd = 0; targetCamInd < cam->bundle.size();
               ++targetCamInd) {

            CameraModel &targetCam = cam->bundle[targetCamInd].cam;
            KeyFrameEntry &targetEntry = targetFrame->frames[targetCamInd];
            if (targetFrame == hostFrame)
              continue;

            SE3 baseToBody = cam->bundle[hostCamInd].thisToBody;
            SE3 bodyToTarget = cam->bundle[targetCamInd].bodyToThis;
            SE3 baseToTarget = bodyToTarget * bodyToWorld[targetInd].inverse() *
                               bodyToWorld[hostInd] * baseToBody;
            pointsTotal++;
            Vec2 curReproj =
                targetCam.map(baseToTarget * (op.depth() * op.dir));
            if (!targetCam.isOnImage(curReproj, PH)) {
              pointsOOB++;
              continue;
            }

            for (int i = 0; i < PS; ++i) {
              const Vec2 &shiftedP =
                  op.p + settings.residualPattern.pattern()[i];

              CHECK(cam->bundle[hostCamInd].cam.isOnImage(op.p, PH))
                  << "Optimized point is not on the image! p = "
                  << op.p.transpose() << "d = " << op.depth()
                  << " min = " << op.minDepth << " max = " << op.maxDepth;

              DirectResidual *newResidual = new DirectResidual(
                  &hostFrame->preKeyFrame->frames[hostCamInd]
                       .internals->interpolator(0),
                  &targetFrame->preKeyFrame->frames[targetCamInd]
                       .internals->interpolator(0),
                  &targetCam, &op, shiftedP, baseToBody, bodyToTarget,
                  hostFrame, targetFrame, settings.depth);

              residuals.push_back(newResidual);

              double gradNorm =
                  hostFrame->preKeyFrame->frames[hostCamInd].gradNorm(
                      toCvPoint(shiftedP));
              const double c = settings.residualWeighting.c;
              double weight = c / std::hypot(c, gradNorm);
              ceres::LossFunction *lossFunc = new ceres::ScaledLoss(
                  new ceres::HuberLoss(settings.intensity.outlierDiff), weight,
                  ceres::Ownership::TAKE_OWNERSHIP);

              problem.AddResidualBlock(
                  new ceres::AutoDiffCostFunction<DirectResidual, 1, 1, 3, 4, 3,
                                                  4, 2, 2>(newResidual),
                  lossFunc, depthParamPtr,
                  bodyToWorld[hostInd].translation().data(),
                  bodyToWorld[hostInd].so3().data(),
                  bodyToWorld[targetInd].translation().data(),
                  bodyToWorld[targetInd].so3().data(),
                  hostFrame->frames[hostCamInd].lightWorldToThis.data,
                  targetFrame->frames[targetCamInd].lightWorldToThis.data);
            }
          }
        }
      }
    }
  }

  CHECK(oldDepthParamsBegin == depthParams.begin());

  LOG(INFO) << "total points = " << numPoints
            << " OOB small = " << numOobSmallDepths
            << " OOB large = " << numOobLargeDepths
            << " bad = " << numBadDepths;

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.linear_solver_ordering = ordering;
  options.max_num_iterations = settings.optimization.maxIterations;
  options.num_threads = settings.threading.numThreads;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  for (int kfInd = 0; kfInd < numKeyFrames; ++kfInd)
    keyFrames[kfInd]->thisToWorld.setValue(bodyToWorld[kfInd]);
  for (auto [op, depthParam] : depthParams) {
    double depth = settings.depth.useMinPlusExpParametrization
                       ? settings.depth.min + std::exp(depthParam)
                       : std::exp(depthParam);
    if (depth > 0)
      op->setDepth(depth);
  }

  LOG(INFO) << summary.FullReport() << std::endl;

  logOobStatistics(residuals, settings);
}

} // namespace mdso
