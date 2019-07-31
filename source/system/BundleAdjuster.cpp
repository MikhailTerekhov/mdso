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

namespace fishdso {

BundleAdjuster::BundleAdjuster(CameraModel *cam,
                               const BundleAdjusterSettings &_settings)
    : cam(cam)
    , firstKeyFrame(nullptr)
    , settings(_settings) {}

bool BundleAdjuster::isOOB(const SE3 &baseToWorld, const SE3 &refToWorld,
                           const OptimizedPoint &baseOP) {
  Vec3 inBase = cam->unmap(baseOP.p).normalized() * baseOP.depth();
  Vec2 reproj = cam->map(refToWorld.inverse() * baseToWorld * inBase);
  return !cam->isOnImage(reproj, settings.residualPattern.height);
}

void BundleAdjuster::addKeyFrame(KeyFrame *keyFrame) {
  if (firstKeyFrame == nullptr)
    firstKeyFrame = keyFrame;
  keyFrames.insert(keyFrame);
}

struct DirectResidual {
  DirectResidual(
      ceres::BiCubicInterpolator<ceres::Grid2D<unsigned char, 1>> *baseFrame,
      ceres::BiCubicInterpolator<ceres::Grid2D<unsigned char, 1>> *refFrame,
      const CameraModel *cam, OptimizedPoint *optimizedPoint, const Vec2 &pos,
      KeyFrame *baseKf, KeyFrame *refKf)
      : cam(cam)
      , baseDirection(cam->unmap(pos).normalized())
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
  ceres::BiCubicInterpolator<ceres::Grid2D<unsigned char, 1>> *refFrame;
  OptimizedPoint *optimizedPoint;
  KeyFrame *baseKf;
  KeyFrame *refKf;
};

void BundleAdjuster::adjust(int maxNumIterations) {
  int pointsTotal = 0, pointsOOB = 0, pointsOutliers = 0;

  KeyFrame *secondKeyFrame = nullptr;
  for (auto kf : keyFrames)
    if (kf != firstKeyFrame) {
      secondKeyFrame = kf;
      break;
    }

  std::shared_ptr<ceres::ParameterBlockOrdering> ordering(
      new ceres::ParameterBlockOrdering());

  ceres::Problem problem;

  for (KeyFrame *keyFrame : keyFrames) {
    problem.AddParameterBlock(keyFrame->thisToWorld.translation().data(), 3);
    problem.AddParameterBlock(keyFrame->thisToWorld.so3().data(), 4,
                              new ceres::EigenQuaternionParameterization());
    auto affLight = keyFrame->lightWorldToThis.data;
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

    ordering->AddElementToGroup(keyFrame->thisToWorld.translation().data(), 1);
    ordering->AddElementToGroup(keyFrame->thisToWorld.so3().data(), 1);
    ordering->AddElementToGroup(affLight, 1);
  }

  problem.SetParameterBlockConstant(
      firstKeyFrame->thisToWorld.translation().data());
  problem.SetParameterBlockConstant(firstKeyFrame->thisToWorld.so3().data());
  problem.SetParameterBlockConstant(firstKeyFrame->lightWorldToThis.data);

  SE3 firstToWorld = firstKeyFrame->thisToWorld;
  SE3 secondToWorld = secondKeyFrame->thisToWorld;
  double radius =
      (secondToWorld.translation() - firstToWorld.translation()).norm();
  Vec3 center = firstToWorld.translation();
  problem.SetParameterization(
      secondKeyFrame->thisToWorld.translation().data(),
      new ceres::AutoDiffLocalParameterization<SphericalPlus, 3, 2>(
          new SphericalPlus(center, radius, secondToWorld.translation())));

  if (settings.bundleAdjuster.fixedRotationOnSecondKF)
    problem.SetParameterBlockConstant(secondKeyFrame->thisToWorld.so3().data());

  if (settings.bundleAdjuster.fixedMotionOnFirstAdjustent &&
      keyFrames.size() == 2) {
    problem.SetParameterBlockConstant(
        secondKeyFrame->thisToWorld.translation().data());
    problem.SetParameterBlockConstant(secondKeyFrame->thisToWorld.so3().data());
    problem.SetParameterBlockConstant(secondKeyFrame->lightWorldToThis.data);
  }

  LOG(INFO) << "points on the first = " << firstKeyFrame->optimizedPoints.size()
            << std::endl;
  LOG(INFO) << "points on the second = "
            << secondKeyFrame->optimizedPoints.size() << std::endl;

  std::map<OptimizedPoint *, std::vector<DirectResidual *>> residualsFor;

  StdVector<Vec2> oobPos;
  StdVector<Vec2> oobKf1;

  for (KeyFrame *baseFrame : keyFrames)
    for (const auto &op : baseFrame->optimizedPoints) {
      problem.AddParameterBlock(&op->logInvDepth, 1);
      problem.SetParameterLowerBound(&op->logInvDepth, 0,
                                     -std::log(settings.depth.max));
      problem.SetParameterUpperBound(&op->logInvDepth, 0,
                                     -std::log(settings.depth.min));

      ordering->AddElementToGroup(&op->logInvDepth, 0);

      for (KeyFrame *refFrame : keyFrames) {
        if (refFrame == baseFrame)
          continue;
        pointsTotal++;
        if (isOOB(baseFrame->thisToWorld, refFrame->thisToWorld, *op)) {
          pointsOOB++;
          if (baseFrame == secondKeyFrame)
            oobPos.push_back(op->p);
          else
            oobKf1.push_back(op->p);
          continue;
        }

        for (int i = 0; i < settings.residualPattern.pattern().size(); ++i) {
          const Vec2 &pos = op->p + settings.residualPattern.pattern()[i];
          DirectResidual *newResidual = new DirectResidual(
              &baseFrame->preKeyFrame->internals->interpolator(0),
              &refFrame->preKeyFrame->internals->interpolator(0), cam, op.get(),
              pos, baseFrame, refFrame);

          double gradNorm = baseFrame->preKeyFrame->gradNorm(toCvPoint(pos));
          const double c = settings.gradWeighting.c;
          double weight = c / std::hypot(c, gradNorm);
          ceres::LossFunction *lossFunc = new ceres::ScaledLoss(
              new ceres::HuberLoss(settings.intencity.outlierDiff), weight,
              ceres::Ownership::TAKE_OWNERSHIP);

          residualsFor[op.get()].push_back(newResidual);
          problem.AddResidualBlock(
              new ceres::AutoDiffCostFunction<DirectResidual, 1, 1, 3, 4, 3, 4,
                                              2, 2>(newResidual),
              lossFunc, &op->logInvDepth,
              baseFrame->thisToWorld.translation().data(),
              baseFrame->thisToWorld.so3().data(),
              refFrame->thisToWorld.translation().data(),
              refFrame->thisToWorld.so3().data(),
              baseFrame->lightWorldToThis.data,
              refFrame->lightWorldToThis.data);
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

  if (secondKeyFrame->optimizedPoints.size() > 0) {
    auto p = std::minmax_element(secondKeyFrame->optimizedPoints.begin(),
                                 secondKeyFrame->optimizedPoints.end(),
                                 [](const auto &op1, const auto &op2) {
                                   return op1->depth() < op2->depth();
                                 });
    LOG(INFO) << "minmax d = " << (*p.first)->depth() << ' '
              << (*p.second)->depth() << std::endl;
  }

  std::vector<double> depthsVec;
  depthsVec.reserve(secondKeyFrame->optimizedPoints.size());
  for (const auto &op : secondKeyFrame->optimizedPoints)
    depthsVec.push_back(op->depth());
  // setDepthColBounds(depthsVec);

  cv::Mat kfDepths = secondKeyFrame->drawDepthedFrame(minDepthCol, maxDepthCol);

  StdVector<Vec2> outliers;
  StdVector<Vec2> badDepth;
  for (const auto &op : secondKeyFrame->optimizedPoints) {
    if (op->state == OptimizedPoint::OOB)
      continue;

    std::vector<double> values =
        reservedVector<double>(residualsFor[op.get()].size());
    for (DirectResidual *res : residualsFor[op.get()]) {
      double value;
      double &logInvDepth = op->logInvDepth;
      KeyFrame *base = res->baseKf;
      KeyFrame *ref = res->refKf;
      if (res->operator()(&logInvDepth, base->thisToWorld.translation().data(),
                          base->thisToWorld.so3().data(),
                          ref->thisToWorld.translation().data(),
                          ref->thisToWorld.so3().data(),
                          base->lightWorldToThis.data,
                          ref->lightWorldToThis.data, &value))
        values.push_back(value);
    }

    if (values.empty()) {
      op->state = OptimizedPoint::OOB;
      continue;
    }

    std::nth_element(values.begin(), values.begin() + values.size() / 2,
                     values.end());
    double median = values[values.size() / 2];
    if (median > settings.intencity.outlierDiff) {
      op->state = OptimizedPoint::OUTLIER;
      outliers.push_back(op->p);
    }
  }
  pointsOutliers = outliers.size();

  LOG(INFO) << "BA results:";
  LOG(INFO) << "total points = " << pointsTotal;
  LOG(INFO) << "OOB points = " << pointsOOB;
  LOG(INFO) << "outlier points = " << pointsOutliers;

  LOG(INFO) << summary.FullReport() << std::endl;
}
} // namespace fishdso
