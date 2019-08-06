#include "system/FrameTracker.h"
#include "PreKeyFrameInternals.h"
#include "output/FrameTrackerObserver.h"
#include "util/defs.h"
#include "util/util.h"
#include <ceres/cubic_interpolation.h>
#include <ceres/problem.h>
#include <chrono>
#include <cmath>

namespace fishdso {

struct PointTrackingResidual {
  PointTrackingResidual(
      Vec3 pos, double baseIntensity, const CameraModel *cam,
      const ceres::BiCubicInterpolator<ceres::Grid2D<unsigned char, 1>>
          *trackedFrame)
      : pos(pos)
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
    AffineLightTransform<T> affLight(affLightP[0], affLightP[1]);

    Vec3t newPos = motion * pos.cast<T>();
    Vec2t newPosProj = cam->map(newPos.data());

    T trackedIntensity;
    trackedFrame->Evaluate(newPosProj[1], newPosProj[0], &trackedIntensity);
    res[0] = affLight(trackedIntensity) - baseIntensity;

    return true;
  }

  Vec3 pos;
  double baseIntensity;
  const CameraModel *cam;
  const ceres::BiCubicInterpolator<ceres::Grid2D<unsigned char, 1>>
      *trackedFrame;
};

FrameTracker::FrameTracker(CameraBundle camPyr[], const DepthedMultiFrame &_baseFrame,
                           std::vector<FrameTrackerObserver *> &observers,
                           const FrameTrackerSettings &_settings)
    : camPyr(camPyr)
    , baseFrame(_baseFrame)
    , observers(observers)
    , settings(_settings) {
  for (FrameTrackerObserver *obs : observers)
    obs->newBaseFrame(baseFrame);
}

FrameTracker::TrackingResult
FrameTracker::trackFrame(const PreKeyFrame &frame,
                         const TrackingResult &coarseTrackingResult) {
  return coarseTrackingResult;
  /*
  for (FrameTrackerObserver *obs : observers)
    obs->startTracking(frame.framePyr);

  SE3 baseToTracked = coarseBaseToTracked;
  AffLight affLight = coarseAffLight;

  for (int i = settings.pyramid.levelNum - 1; i >= 0; --i) {
    LOG(INFO) << "track level #" << i << std::endl;
    std::tie(baseToTracked, affLight) = trackPyrLevel(
        camPyr[i], baseFrame->images[i], baseFrame->depths[i],
        frame.framePyr.images[i], *frame.internals, baseToTracked, affLight, i);
  }

  // cv::waitKey();

  return {baseToTracked, affLight};
  */
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
  return coarseTrackingResult;
  /*
  SE3 baseToTracked = coarseBaseToTracked;
  AffLight affLight = coarseAffLight;

  cv::Mat1b resMask(baseImg.size(), CV_WHITE_BYTE);

  const ceres::BiCubicInterpolator<ceres::Grid2D<unsigned char, 1>>
      &trackedFrame = internals.interpolator(pyrLevel);

  ceres::Problem problem;

  problem.AddParameterBlock(baseToTracked.so3().data(), 4,
                            new ceres::EigenQuaternionParameterization());
  problem.AddParameterBlock(baseToTracked.translation().data(), 3);

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

  std::vector<double> pixUsed;

  double pnt[2] = {0.0, 0.0};

  std::vector<const PointTrackingResidual *> residuals;
  StdVector<Vec2> gotOutside;

  int pntTotal = 0;

  for (int y = 0; y < baseImg.rows; ++y)
    for (int x = 0; x < baseImg.cols; ++x)
      if (baseDepths(y, x) > 0) {
        if (!resMask(y, x))
          continue;
        ++pntTotal;
        pnt[0] = x;
        pnt[1] = y;

        Vec3 pos = cam.unmap(pnt).normalized() * baseDepths(y, x);
        if (!isPointTrackable(cam, pos, coarseBaseToTracked)) {
          gotOutside.push_back(Vec2(x, y));
          continue;
        }

        pixUsed.push_back(baseImg(y, x));

        ceres::LossFunction *lossFunc = nullptr;
        if (settings.frameTracker.useGradWeighting) {
          double gradNorm = gradNormAt(baseImg, cv::Point(x, y));
          double c = settings.gradWeighting.c;
          double weight = c / std::hypot(c, gradNorm);
          lossFunc = new ceres::ScaledLoss(
              new ceres::HuberLoss(settings.intencity.outlierDiff), weight,
              ceres::Ownership::TAKE_OWNERSHIP);
        } else
          lossFunc = new ceres::HuberLoss(settings.intencity.outlierDiff);

        auto newResidual = new PointTrackingResidual(
            pos, static_cast<double>(baseImg(y, x)), &cam, &trackedFrame);
        residuals.push_back(newResidual);
        ceres::CostFunction *newCostFunc =
            new ceres::AutoDiffCostFunction<PointTrackingResidual, 1, 4, 3, 2>(
                newResidual);
        problem.AddResidualBlock(
            newCostFunc, lossFunc, baseToTracked.so3().data(),
            baseToTracked.translation().data(), affLight.data);
      }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.num_threads = settings.threading.numThreads;
  // options.minimizer_progress_to_stdout = true;
  // options.max_num_iterations = 10;
  ceres::Solver::Summary summary;

  ceres::Solve(options, &problem, &summary);
  LOG(INFO) << "time (mcs) = " << summary.total_time_in_seconds * 1000
            << std::endl;

  LOG(INFO) << summary.BriefReport() << std::endl;

  StdVector<std::pair<Vec2, double>> pointResiduals;
  pointResiduals.reserve(residuals.size());

  for (auto res : residuals) {
    double eval = -1;
    (*res)(baseToTracked.unit_quaternion().coeffs().data(),
           baseToTracked.translation().data(), affLight.data, &eval);
    Vec2 onTracked = cam.map(baseToTracked * res->pos);
    pointResiduals.push_back(std::pair(onTracked, eval));
  }

  for (FrameTrackerObserver *obs : observers)
    obs->levelTracked(pyrLevel, baseToTracked, affLight, pointResiduals);

  return {baseToTracked, affLight};
  */
}

} // namespace fishdso
