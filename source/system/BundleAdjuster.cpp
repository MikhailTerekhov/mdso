#include "system/BundleAdjuster.h"
#include "PreKeyFrameInternals.h"
#include "system/AffineLightTransform.h"
#include "system/SphericalPlus.h"
#include "util/defs.h"
#include "util/geometry.h"
#include "util/util.h"
#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>
#include <ceres/local_parameterization.h>

#define PS (settings.residualPattern.pattern().size())
#define PH (settings.residualPattern.height)

namespace mdso {

BundleAdjuster::BundleAdjuster(CameraBundle *cam, KeyFrame *keyFrames[],
                               int size,
                               const BundleAdjusterSettings &_settings)
    : cam(cam)
    , keyFrames(keyFrames)
    , size(size)
    , settings(_settings) {
  CHECK(size >= 2);
  CHECK(cam->bundle.size() == 1) << "Multicamera case is NIY";
}

struct DirectResidual {
  DirectResidual(
      ceres::BiCubicInterpolator<ceres::Grid2D<unsigned char, 1>> *baseFrame,
      ceres::BiCubicInterpolator<ceres::Grid2D<unsigned char, 1>> *refFrame,
      const CameraModel *cam, OptimizedPoint *optimizedPoint, const Vec2 &pos,
      const SE3 &baseToBody, const SE3 &bodyToRef, KeyFrame *baseKf,
      KeyFrame *refKf)
      : cam(cam)
      , baseDirection(cam->unmap(pos).normalized())
      , baseToBody(baseToBody)
      , bodyToRef(bodyToRef)
      , refFrame(refFrame)
      , optimizedPoint(optimizedPoint)
      , baseKf(baseKf)
      , refKf(refKf) {
    baseFrame->Evaluate(pos[1], pos[0], &baseIntencity);
  }

  template <typename T>
  bool operator()(const T *const logInvDepthP, const T *const baseTransP,
                  const T *const baseRotP, const T *const refTransP,
                  const T *const refRotP, const T *const baseAffP,
                  const T *const refAffP, T *res) const {
    typedef Eigen::Matrix<T, 2, 1> Vec2t;
    typedef Eigen::Matrix<T, 3, 1> Vec3t;
    typedef Eigen::Matrix<T, 3, 3> Mat33t;
    typedef Eigen::Quaternion<T> Quatt;
    typedef Sophus::SE3<T> SE3t;

    Eigen::Map<const Vec3t> baseTransM(baseTransP);
    Vec3t baseTrans(baseTransM);
    Eigen::Map<const Quatt> baseRotM(baseRotP);
    Quatt baseRot(baseRotM);
    SE3t baseToWorld(baseRot, baseTrans);

    Eigen::Map<const Vec3t> refTransM(refTransP);
    Vec3t refTrans(refTransM);
    Eigen::Map<const Quatt> refRotM(refRotP);
    Quatt refRot(refRotM);
    SE3t refToWorld(refRot, refTrans);

    const T *baseAffLightP = baseAffP;
    AffineLightTransform<T> baseAffLight(baseAffLightP[0], baseAffLightP[1]);

    const T *refAffLightP = refAffP;
    AffineLightTransform<T> refAffLight(refAffLightP[0], refAffLightP[1]);

    AffineLightTransform<T>::normalizeMultiplier(refAffLight, baseAffLight);

    T depth = ceres::exp(-(*logInvDepthP));
    Vec3t refPos = (refToWorld.inverse() * baseToWorld) *
                   (baseDirection.cast<T>() * depth);
    Vec2t refPosMapped = cam->map(refPos.data()).template cast<T>();
    T trackedIntensity;
    refFrame->Evaluate(refPosMapped[1], refPosMapped[0], &trackedIntensity);
    *res = refAffLight(trackedIntensity) - baseAffLight(T(baseIntencity));

    return true;
  }

  const CameraModel *cam;
  Vec3 baseDirection;
  double baseIntencity;
  SE3 baseToBody, bodyToRef;
  ceres::BiCubicInterpolator<ceres::Grid2D<unsigned char, 1>> *refFrame;
  OptimizedPoint *optimizedPoint;
  KeyFrame *baseKf;
  KeyFrame *refKf;
};

void BundleAdjuster::adjust(int maxNumIterations) {
  int pointsTotal = 0, pointsOOB = 0;

  CameraModel &camera = cam->bundle[0].cam;

  std::shared_ptr<ceres::ParameterBlockOrdering> ordering(
      new ceres::ParameterBlockOrdering());

  ceres::Problem problem;

  for (int i = 0; i < size; ++i) {
    KeyFrame *keyFrame = keyFrames[i];

    problem.AddParameterBlock(keyFrame->thisToWorld.translation().data(), 3);
    problem.AddParameterBlock(keyFrame->thisToWorld.so3().data(), 4,
                              new ceres::EigenQuaternionParameterization());

    ordering->AddElementToGroup(keyFrame->thisToWorld.translation().data(), 1);
    ordering->AddElementToGroup(keyFrame->thisToWorld.so3().data(), 1);

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

  problem.SetParameterBlockConstant(
      keyFrames[0]->thisToWorld.translation().data());
  problem.SetParameterBlockConstant(keyFrames[0]->thisToWorld.so3().data());
  for (KeyFrameEntry &entry : keyFrames[0]->frames)
    problem.SetParameterBlockConstant(entry.lightWorldToThis.data);

  SE3 firstToWorld = keyFrames[0]->thisToWorld;
  SE3 secondToWorld = keyFrames[1]->thisToWorld;
  double radius =
      (secondToWorld.translation() - firstToWorld.translation()).norm();
  Vec3 center = firstToWorld.translation();
  if (radius > settings.bundleAdjuster.minFirstToSecondRadius)
    problem.SetParameterization(
        keyFrames[1]->thisToWorld.translation().data(),
        new ceres::AutoDiffLocalParameterization<SphericalPlus, 3, 2>(
            new SphericalPlus(center, radius, secondToWorld.translation())));
  else
    problem.SetParameterBlockConstant(
        keyFrames[1]->thisToWorld.translation().data());

  if (settings.bundleAdjuster.fixedRotationOnSecondKF)
    problem.SetParameterBlockConstant(keyFrames[1]->thisToWorld.so3().data());

  if (settings.bundleAdjuster.fixedMotionOnFirstAdjustent && size == 2) {
    problem.SetParameterBlockConstant(
        keyFrames[1]->thisToWorld.translation().data());
    problem.SetParameterBlockConstant(keyFrames[1]->thisToWorld.so3().data());
  }

  for (int indBase = 0; indBase < size; ++indBase) {
    KeyFrame *baseFrame = keyFrames[indBase];
    for (auto &op : baseFrame->frames[0].optimizedPoints) {
      problem.AddParameterBlock(&op.logInvDepth, 1);
      problem.SetParameterLowerBound(&op.logInvDepth, 0,
                                     -std::log(settings.depth.max));
      problem.SetParameterUpperBound(&op.logInvDepth, 0,
                                     -std::log(settings.depth.min));

      ordering->AddElementToGroup(&op.logInvDepth, 0);

      for (int indRef = 0; indRef < size; ++indRef) {
        KeyFrame *refFrame = keyFrames[indRef];
        if (refFrame == baseFrame)
          continue;

        // TODO inds in multicamera
        SE3 baseToBody = cam->bundle[0].thisToBody;
        SE3 bodyToRef = cam->bundle[0].bodyToThis;
        SE3 baseToRef = bodyToRef * refFrame->thisToWorld.inverse() *
                        baseFrame->thisToWorld * baseToBody;
        pointsTotal++;
        if (!camera.isOnImage(camera.map(baseToRef * (op.depth() * op.dir)),
                              PH)) {
          pointsOOB++;
          continue;
        }

        for (int i = 0; i < PS; ++i) {
          const Vec2 &pos = op.p + settings.residualPattern.pattern()[i];
          // TODO inds in multicamera
          DirectResidual *newResidual = new DirectResidual(
              &baseFrame->preKeyFrame->internals->frames[0].interpolator(0),
              &refFrame->preKeyFrame->internals->frames[0].interpolator(0),
              &camera, &op, pos, baseToBody, bodyToRef, baseFrame, refFrame);

          // TODO inds in multicamera
          double gradNorm =
              baseFrame->preKeyFrame->frames[0].gradNorm(toCvPoint(pos));
          const double c = settings.gradWeighting.c;
          double weight = c / std::hypot(c, gradNorm);
          ceres::LossFunction *lossFunc = new ceres::ScaledLoss(
              new ceres::HuberLoss(settings.intencity.outlierDiff), weight,
              ceres::Ownership::TAKE_OWNERSHIP);

          // TODO inds in multicamera
          problem.AddResidualBlock(
              new ceres::AutoDiffCostFunction<DirectResidual, 1, 1, 3, 4, 3, 4,
                                              2, 2>(newResidual),
              lossFunc, &op.logInvDepth,
              baseFrame->thisToWorld.translation().data(),
              baseFrame->thisToWorld.so3().data(),
              refFrame->thisToWorld.translation().data(),
              refFrame->thisToWorld.so3().data(),
              baseFrame->frames[0].lightWorldToThis.data,
              refFrame->frames[0].lightWorldToThis.data);
        }
      }
    }
  }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.linear_solver_ordering = ordering;
  // options.minimizer_progress_to_stdout = true;
  options.max_num_iterations = maxNumIterations;
  options.num_threads = settings.threading.numThreads;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  LOG(INFO) << summary.FullReport() << std::endl;
}
} // namespace mdso
