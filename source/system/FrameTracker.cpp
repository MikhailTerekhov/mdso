#include "system/FrameTracker.h"
#include "PreKeyFrameInternals.h"
#include "output/FrameTrackerObserver.h"
#include "util/defs.h"
#include "util/util.h"
#include <ceres/cubic_interpolation.h>
#include <ceres/problem.h>
#include <chrono>
#include <cmath>

namespace mdso {

struct PointTrackingResidual {
  PointTrackingResidual(
      Vec3 pos, double baseIntensity, const SE3 &baseToBody,
      const SE3 &bodyToRef, const CameraModel *cam,
      const ceres::BiCubicInterpolator<ceres::Grid2D<unsigned char, 1>>
          *trackedFrame)
      : pos(pos)
      , baseToBody(baseToBody)
      , bodyToRef(bodyToRef)
      , baseIntensity(baseIntensity)
      , cam(cam)
      , trackedFrame(trackedFrame) {}

  template <typename T>
  bool operator()(const T *const rotP, const T *const transP,
                  const T *const affLightP, T *res) const {
    typedef Eigen::Matrix<T, 2, 1> Vec2t;
    typedef Eigen::Matrix<T, 3, 1> Vec3t;
    typedef Eigen::Matrix<T, 3, 3> Mat33t;
    typedef Eigen::Quaternion<T> Quatt;
    typedef Sophus::SE3<T> SE3t;

    Eigen::Map<const Vec3t> transM(transP);
    Vec3t trans(transM);
    Eigen::Map<const Quatt> rotM(rotP);
    Quatt rot(rotM);
    SE3t motion(rot, trans);
    SE3t baseToBodyT = baseToBody.cast<T>();
    SE3t bodyToRefT = bodyToRef.cast<T>();

    AffineLightTransform<T> affLight(affLightP[0], affLightP[1]);

    Vec3t newPos = bodyToRefT * motion * baseToBodyT * pos.cast<T>();
    Vec2t newPosProj = cam->map(newPos.data());

    T trackedIntensity;
    trackedFrame->Evaluate(newPosProj[1], newPosProj[0], &trackedIntensity);
    res[0] = affLight(trackedIntensity) - baseIntensity;

    return true;
  }

  Vec3 pos;
  SE3 baseToBody, bodyToRef;
  double baseIntensity;
  const CameraModel *cam;
  const ceres::BiCubicInterpolator<ceres::Grid2D<unsigned char, 1>>
      *trackedFrame;
};

FrameTracker::TrackingResult::TrackingResult(int camNumber)
    : lightBaseToTracked(camNumber) {}

FrameTracker::FrameTracker(CameraBundle camPyr[],
                           const DepthedMultiFrame &_baseFrame,
                           std::vector<FrameTrackerObserver *> &observers,
                           const FrameTrackerSettings &_settings)
    : camPyr(camPyr)
    , baseFrame(_baseFrame)
    , observers(observers)
    , settings(_settings) {
  CHECK(camPyr[0].bundle.size() == 1) << "Multicamera case is NIY";
  for (FrameTrackerObserver *obs : observers)
    obs->newBaseFrame(baseFrame);
}

FrameTracker::TrackingResult
FrameTracker::trackFrame(const PreKeyFrame &frame,
                         const TrackingResult &coarseTrackingResult) {
  for (FrameTrackerObserver *obs : observers)
    obs->startTracking(frame);

  TrackingResult result = coarseTrackingResult;

  for (int i = settings.pyramid.levelNum() - 1; i >= 0; --i) {
    LOG(INFO) << "track level #" << i << std::endl;
    result = trackPyrLevel(frame, result, i);
  }

  // cv::waitKey();

  return result;
}

bool isPointTrackable(const CameraModel &cam, const Vec3 &basePos,
                      const SE3 &coarseBaseToCur) {
  Vec3 coarseCurPos = coarseBaseToCur * basePos;
  Vec2 coarseCurOnImg = cam.map(coarseCurPos);
  return cam.isOnImage(coarseCurOnImg, 0);
}

FrameTracker::TrackingResult
FrameTracker::trackPyrLevel(const PreKeyFrame &frame,
                            const TrackingResult &coarseTrackingResult,
                            int pyrLevel) {
  CameraBundle &cam = camPyr[pyrLevel];

  CHECK(cam.bundle.size() == 1) << "Multicamera case is NIY";

  TrackingResult result = coarseTrackingResult;
  // const ceres::BiCubicInterpolator<ceres::Grid2D<unsigned char, 1>>
  // &trackedFrame = internals.interpolator(pyrLevel);

  ceres::Problem problem;

  problem.AddParameterBlock(result.baseToTracked.so3().data(), 4,
                            new ceres::EigenQuaternionParameterization());
  problem.AddParameterBlock(result.baseToTracked.translation().data(), 3);

  for (auto &affLight : result.lightBaseToTracked) {
    problem.AddParameterBlock(affLight.data, 2);
    problem.SetParameterLowerBound(affLight.data, 0,
                                   settings.affineLight.minAffineLightA);
    problem.SetParameterUpperBound(affLight.data, 0,
                                   settings.affineLight.maxAffineLightA);
    problem.SetParameterLowerBound(affLight.data, 1,
                                   settings.affineLight.minAffineLightB);
    problem.SetParameterUpperBound(affLight.data, 1,
                                   settings.affineLight.maxAffineLightB);
    if (!settings.affineLight.optimizeAffineLight)
      problem.SetParameterBlockConstant(affLight.data);
  }

  std::vector<const PointTrackingResidual *> residuals;
  StdVector<Vec2> gotOutside;

  int pntTotal = 0;

  for (int y = 0; y < baseFrame[0].depths[pyrLevel].rows; ++y)
    for (int x = 0; x < baseFrame[0].depths[pyrLevel].cols; ++x) {
      double baseDepth = baseFrame[0].depths[pyrLevel](y, x);
      if (baseDepth <= 0)
        continue;

      ++pntTotal;

      Vec3 pos = cam.bundle[0].cam.unmap(Vec2(x, y)).normalized() * baseDepth;
      if (!isPointTrackable(cam.bundle[0].cam, pos, result.baseToTracked)) {
        gotOutside.push_back(Vec2(x, y));
        continue;
      }

      ceres::LossFunction *lossFunc = nullptr;
      if (settings.frameTracker.useGradWeighting) {
        double gradNorm = frame.frames[0].gradNorm(y, x);
        double c = settings.gradWeighting.c;
        double weight = c / std::hypot(c, gradNorm);
        lossFunc = new ceres::ScaledLoss(
            new ceres::HuberLoss(settings.intencity.outlierDiff), weight,
            ceres::Ownership::TAKE_OWNERSHIP);
      } else
        lossFunc = new ceres::HuberLoss(settings.intencity.outlierDiff);

      double intencity = baseFrame[0].images[pyrLevel](y, x);

      // TODO inds in multicamera
      auto newResidual = new PointTrackingResidual(
          pos, intencity, cam.bundle[0].bodyToThis, cam.bundle[0].thisToBody,
          &cam.bundle[0].cam,
          &frame.internals->frames[0].interpolator(pyrLevel));
      residuals.push_back(newResidual);
      ceres::CostFunction *newCostFunc =
          new ceres::AutoDiffCostFunction<PointTrackingResidual, 1, 4, 3, 2>(
              newResidual);
      problem.AddResidualBlock(newCostFunc, lossFunc,
                               result.baseToTracked.so3().data(),
                               result.baseToTracked.translation().data(),
                               result.lightBaseToTracked[0].data);
    }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.num_threads = settings.threading.numThreads;
  // options.minimizer_progress_to_stdout = true;
  // options.max_num_iterations = 10;
  ceres::Solver::Summary summary;

  ceres::Solve(options, &problem, &summary);
  LOG(INFO) << "time (ms) = " << summary.total_time_in_seconds * 1000
            << std::endl;

  LOG(INFO) << summary.BriefReport() << std::endl;

  StdVector<std::pair<Vec2, double>> pointResiduals;
  pointResiduals.reserve(residuals.size());

  for (auto res : residuals) {
    double eval = -1;
    (*res)(result.baseToTracked.unit_quaternion().coeffs().data(),
           result.baseToTracked.translation().data(),
           result.lightBaseToTracked[0].data, &eval);
    Vec2 onTracked = cam.bundle[0].cam.map(result.baseToTracked * res->pos);
    pointResiduals.push_back(std::pair(onTracked, eval));
  }

  for (FrameTrackerObserver *obs : observers)
    obs->levelTracked(pyrLevel, result, pointResiduals.data(),
                      pointResiduals.size());

  return result;
}

} // namespace mdso
