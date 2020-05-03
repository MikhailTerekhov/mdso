#include "internal/optimize/EnergyFunctionCeres.h"
#include "optimize/SphericalPlus.h"
#include "system/KeyFrame.h"
#include "system/Reprojector.h"

#define PS (settings.residualPattern.pattern().size())
#define PH (settings.residualPattern.height)

namespace mdso::optimize {

EnergyFunctionCeres::Residual::Residual(CameraBundle *cameraBundle,
                                        KeyFrameEntry *hostKfEntry,
                                        KeyFrameEntry *targetKfEntry,
                                        OptimizedPoint *optimizedPoint,
                                        int numInPattern,
                                        const BundleAdjusterSettings &settings)
    : cameraBundle(cameraBundle)
    , hostCam(&cameraBundle->bundle[hostKfEntry->ind].cam)
    , targetCam(&cameraBundle->bundle[targetKfEntry->ind].cam)
    , hostKf(hostKfEntry->host)
    , targetKf(targetKfEntry->host)
    , hostImage(&hostKf->preKeyFrame->frames[hostKfEntry->ind]
        .internals->interpolator(0))
    , targetImage(&targetKf->preKeyFrame->frames[targetKfEntry->ind]
        .internals->interpolator(0))
    , hostFrameToBody(cameraBundle->bundle[hostKfEntry->ind].thisToBody)
    , targetBodyToFrame(cameraBundle->bundle[targetKfEntry->ind].bodyToThis)
    , optimizedPoint(optimizedPoint)
    , hostDirection(optimizedPoint->dir)
    , numInPattern(numInPattern)
    , settings(settings) {
  Vec2 pixelPos =
      optimizedPoint->p + settings.residualPattern.pattern()[numInPattern];
  hostImage->Evaluate(pixelPos[1], pixelPos[0], &hostIntensity);
  Vec3 hostPatternDir = hostCam->unmap(pixelPos).normalized();
  SE3 hostToWorld = hostKf->thisToWorld();
  SE3 targetToWorld = targetKf->thisToWorld();
  double depth = optimizedPoint->depth();
  SE3 hostFrameToTargetFrame = targetBodyToFrame * targetToWorld.inverse() *
                               hostToWorld * hostFrameToBody;
  Vec3 targetPatternDir, targetPointDir;
  if (depth > settings.depth.max) {
    targetPatternDir = hostFrameToTargetFrame.so3() * hostPatternDir;
    targetPointDir = hostFrameToTargetFrame.so3() * hostDirection;
  } else {
    targetPatternDir = hostFrameToTargetFrame * (depth * hostPatternDir);
    targetPointDir = hostFrameToTargetFrame * (depth * hostDirection);
  }
  targetPatternDelta =
      targetCam->map(targetPatternDir) - targetCam->map(targetPointDir);
}

void logOobStatistics(
    const std::vector<EnergyFunctionCeres::Residual *> &residuals,
    const BundleAdjusterSettings &settings) {
  int numOob = 0, numOobBigDepth = 0;
  for (const auto *res : residuals) {
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

EnergyFunctionCeres::EnergyFunctionCeres(
    KeyFrame *newKeyFrames[], int numKeyFrames,
    const BundleAdjusterSettings &newSettings)
    : keyFrames(newKeyFrames, newKeyFrames + numKeyFrames)
    , ordering(new ceres::ParameterBlockOrdering())
    , settings(newSettings) {
  CHECK_GE(numKeyFrames, 2);

  CameraBundle *cam = keyFrames[0]->preKeyFrame->cam;
  pointParams.reserve(Settings::max_maxOptimizedPoints);
  auto oldDepthParamsBegin = pointParams.begin();

  int pointsTotal = 0, pointsOOB = 0;

  bodyToWorld.reserve(numKeyFrames);
  for (int i = 0; i < numKeyFrames; ++i) {
    KeyFrame *keyFrame = keyFrames[i];
    bodyToWorld.push_back(keyFrame->thisToWorld());

    mProblem.AddParameterBlock(bodyToWorld.back().translation().data(), 3);
    mProblem.AddParameterBlock(bodyToWorld.back().so3().data(), 4,
                               new ceres::EigenQuaternionParameterization());

    ordering->AddElementToGroup(bodyToWorld.back().translation().data(), 1);
    ordering->AddElementToGroup(bodyToWorld.back().so3().data(), 1);

    for (KeyFrameEntry &entry : keyFrame->frames) {
      double *affLight = entry.lightWorldToThis.data;
      mProblem.AddParameterBlock(affLight, 2);
      mProblem.SetParameterLowerBound(affLight, 0,
                                      settings.affineLight.minAffineLightA);
      mProblem.SetParameterUpperBound(affLight, 0,
                                      settings.affineLight.maxAffineLightA);
      mProblem.SetParameterLowerBound(affLight, 1,
                                      settings.affineLight.minAffineLightB);
      mProblem.SetParameterUpperBound(affLight, 1,
                                      settings.affineLight.maxAffineLightB);
      if (!settings.affineLight.optimizeAffineLight)
        mProblem.SetParameterBlockConstant(affLight);
      ordering->AddElementToGroup(affLight, 1);
    }
  }

  mProblem.SetParameterBlockConstant(bodyToWorld[0].translation().data());
  mProblem.SetParameterBlockConstant(bodyToWorld[0].so3().data());
  for (KeyFrameEntry &entry : keyFrames[0]->frames)
    mProblem.SetParameterBlockConstant(entry.lightWorldToThis.data);

  SE3 firstToWorld = bodyToWorld[0];
  SE3 secondToWorld = bodyToWorld[1];
  double radius =
      (secondToWorld.translation() - firstToWorld.translation()).norm();
  Vec3 center = firstToWorld.translation();
  if (radius > settings.optimization.minFirstToSecondRadius)
    mProblem.SetParameterization(
        bodyToWorld[1].translation().data(),
        new ceres::AutoDiffLocalParameterization<SphericalPlus, 3, 2>(
            new SphericalPlus(center, radius, secondToWorld.translation())));
  else {
    mProblem.SetParameterBlockConstant(bodyToWorld[1].translation().data());
    LOG(WARNING) << "FirstToSecond distance too small, spherical "
                    "parametrization was not applied";
  }

  if (settings.optimization.fixedRotationOnSecondKF)
    mProblem.SetParameterBlockConstant(bodyToWorld[1].so3().data());

  if (settings.optimization.fixedMotionOnFirstAdjustent && numKeyFrames == 2) {
    mProblem.SetParameterBlockConstant(bodyToWorld[1].translation().data());
    mProblem.SetParameterBlockConstant(bodyToWorld[1].so3().data());
  }

  int numPoints = 0;
  int numBadDepths = 0, numOobSmallDepths = 0, numOobLargeDepths = 0;
  Array2d<std::vector<int>> globalPointInds(
      boost::extents[keyFrames.size()][cam->bundle.size()]);
  for (int frameInd = 0; frameInd < numKeyFrames; ++frameInd) {
    KeyFrame *keyFrame = keyFrames[frameInd];
    for (int camInd = 0; camInd < cam->bundle.size(); ++camInd) {
      KeyFrameEntry &hostEntry = keyFrame->frames[camInd];
      globalPointInds[frameInd][camInd].resize(hostEntry.optimizedPoints.size(),
                                               -1);
      for (int pointInd = 0; pointInd < hostEntry.optimizedPoints.size();
           ++pointInd) {
        OptimizedPoint &op = hostEntry.optimizedPoints[pointInd];
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

        globalPointInds[frameInd][camInd][pointInd] = pointParams.size();
        pointParams.push_back({std::log(op.depth()), &op});

        double *depthParamPtr = &pointParams.back().depthParam;
        mProblem.AddParameterBlock(depthParamPtr, 1);
        if (settings.depth.useMinPlusExpParametrization)
          *depthParamPtr = std::log(op.depth() - settings.depth.min);
        else if (settings.depth.setMinBound)
          mProblem.SetParameterLowerBound(depthParamPtr, 0,
                                          std::log(settings.depth.min));
        if (settings.depth.setMaxBound) {
          if (settings.depth.useMinPlusExpParametrization)
            mProblem.SetParameterUpperBound(
                depthParamPtr, 0,
                std::log(settings.depth.max - settings.depth.min));
          else
            mProblem.SetParameterUpperBound(depthParamPtr, 0,
                                            std::log(settings.depth.max));
        }

        ordering->AddElementToGroup(depthParamPtr, 0);
      }
    }
  }

  CHECK(oldDepthParamsBegin == pointParams.begin());

  for (int targetInd = 0; targetInd < numKeyFrames; ++targetInd) {
    KeyFrame *targetFrame = keyFrames[targetInd];
    Reprojector<OptimizedPoint> reprojector(keyFrames.data(), keyFrames.size(),
                                            keyFrames[targetInd]->thisToWorld(),
                                            settings.depth, PH);
    reprojector.setSkippedFrame(targetInd);
    StdVector<Reprojection> reprojections = reprojector.reproject();
    for (const auto &repr : reprojections) {
      int targetCamInd = repr.targetCamInd, hostInd = repr.hostInd,
          hostCamInd = repr.hostCamInd, pointInd = repr.pointInd;
      CHECK_NE(hostInd, targetInd);
      CameraModel &targetCam = cam->bundle[targetCamInd].cam;
      KeyFrame *hostFrame = keyFrames[hostInd];
      OptimizedPoint &op =
          keyFrames[hostInd]->frames[hostCamInd].optimizedPoints[pointInd];
      int globalPointInd = globalPointInds[hostInd][hostCamInd][pointInd];
      if (globalPointInd == -1)
        continue;
      CHECK_GE(globalPointInd, 0);
      CHECK_LT(globalPointInd, pointParams.size());
      CHECK_EQ(pointParams[globalPointInd].op, &op);
      CHECK(targetCam.isOnImage(repr.reprojected, PH));
      CHECK(cam->bundle[hostCamInd].cam.isOnImage(op.p, PH))
      << "Optimized point is not on the image! p = " << op.p.transpose()
      << "d = " << op.depth() << " min = " << op.minDepth
      << " max = " << op.maxDepth;
      double *depthParamPtr = &pointParams[globalPointInd].depthParam;

      for (int i = 0; i < PS; ++i) {
        Residual *newResidual =
            new Residual(cam, &hostFrame->frames[hostCamInd],
                         &targetFrame->frames[targetCamInd], &op, i, settings);

        residuals.push_back(newResidual);

        const Vec2 &shiftedP = op.p + settings.residualPattern.pattern()[i];
        double gradNorm = hostFrame->preKeyFrame->frames[hostCamInd].gradNorm(
            toCvPoint(shiftedP));
        const double c = settings.residualWeighting.c;
        double weight = settings.residualWeighting.useGradientWeighting
                        ? c / std::hypot(c, gradNorm)
                        : 1;
        ceres::LossFunction *lossFunc = new ceres::ScaledLoss(
            new ceres::HuberLoss(settings.intensity.outlierDiff), weight,
            ceres::Ownership::TAKE_OWNERSHIP);

        mProblem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<Residual, 1, 1, 3, 4, 3, 4, 2, 2>(
                newResidual),
            lossFunc, depthParamPtr, bodyToWorld[hostInd].translation().data(),
            bodyToWorld[hostInd].so3().data(),
            bodyToWorld[targetInd].translation().data(),
            bodyToWorld[targetInd].so3().data(),
            hostFrame->frames[hostCamInd].lightWorldToThis.data,
            targetFrame->frames[targetCamInd].lightWorldToThis.data);
      }
    }
  }

  LOG(INFO) << "total points = " << numPoints
            << " OOB small = " << numOobSmallDepths
            << " OOB large = " << numOobLargeDepths
            << " bad = " << numBadDepths;

  logOobStatistics(residuals, settings);
}

void EnergyFunctionCeres::optimize() {
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.linear_solver_ordering = ordering;
  options.max_num_iterations = settings.optimization.maxIterations;
  options.num_threads = settings.threading.numThreads;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem(), &summary);
  LOG(INFO) << summary.FullReport() << std::endl;
}

void EnergyFunctionCeres::applyParameterUpdate() {
  for (int kfInd = 0; kfInd < keyFrames.size(); ++kfInd)
    keyFrames[kfInd]->thisToWorld.setValue(bodyToWorld[kfInd]);
  for (auto [depthParam, op] : pointParams) {
    double depth = settings.depth.useMinPlusExpParametrization
                   ? settings.depth.min + std::exp(depthParam)
                   : std::exp(depthParam);
    if (depth > 0)
      op->setDepth(depth);
  }
}

ceres::Problem &EnergyFunctionCeres::problem() { return mProblem; }

const EnergyFunctionCeres::Residual &
EnergyFunctionCeres::residual(int residualInd) const {
  CHECK_GE(residualInd, 0);
  CHECK_LT(residualInd, residuals.size());
  return *residuals[residualInd];
}
std::shared_ptr<ceres::ParameterBlockOrdering>
EnergyFunctionCeres::parameterBlockOrdering() const {
  return ordering;
}

} // namespace mdso::optimize