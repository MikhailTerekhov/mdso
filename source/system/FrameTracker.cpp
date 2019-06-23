#include "system/FrameTracker.h"
#include "output/FrameTrackerObserver.h"
#include "util/defs.h"
#include "util/util.h"
#include <ceres/cubic_interpolation.h>
#include <ceres/problem.h>
#include <chrono>
#include <cmath>

namespace fishdso {

FrameTracker::FrameTracker(const StdVector<CameraModel> &camPyr,
                           std::unique_ptr<DepthedImagePyramid> baseFrame,
                           const std::vector<FrameTrackerObserver *> &observers,
                           const FrameTrackerSettings &_settings)
    : residualsImg(_settings.pyramid.levelNum)
    , lastRmse(INF)
    , camPyr(camPyr)
    , baseFrame(std::move(baseFrame))
    , displayWidth(camPyr[1].getWidth())
    , displayHeight(camPyr[1].getHeight())
    , observers(observers)
    , settings(_settings) {}

void FrameTracker::addObserver(FrameTrackerObserver *observer) {
  observers.push_back(observer);
}

std::pair<SE3, AffineLightTransform<double>>
FrameTracker::trackFrame(const ImagePyramid &frame, const SE3 &coarseMotion,
                         const AffineLightTransform<double> &coarseAffLight) {
  for (FrameTrackerObserver *obs : observers)
    obs->startTracking(frame);

  SE3 motion = coarseMotion;
  AffineLightTransform<double> affLight;

  for (int i = settings.pyramid.levelNum - 1; i >= 0; --i) {
    LOG(INFO) << "track level #" << i << std::endl;
    std::tie(motion, affLight) =
        trackPyrLevel(camPyr[i], baseFrame->images[i], baseFrame->depthPyr[i],
                      frame.images[i], motion, affLight, i);
  }

  // cv::waitKey();

  return {motion, affLight};
}

bool isPointTrackable(const CameraModel &cam, const Vec3 &basePos,
                      const SE3 &coarseBaseToCur) {
  Vec3 coarseCurPos = coarseBaseToCur * basePos;
  Vec2 coarseCurOnImg = cam.map(coarseCurPos);
  return cam.isOnImage(coarseCurOnImg, 0);
}

std::pair<SE3, AffineLightTransform<double>> FrameTracker::trackPyrLevel(
    const CameraModel &cam, const cv::Mat1b &baseImg,
    const StdVector<DepthedImagePyramid::Point> &depthLayer,
    const cv::Mat1b &trackedImg, const SE3 &coarseMotion,
    const AffineLightTransform<double> &coarseAffLight, int pyrLevel) {
  SE3 motion = coarseMotion;
  AffineLightTransform<double> affLight = coarseAffLight;

  cv::Size displSz = cv::Size(displayHeight, displayWidth);
  cv::Mat1b resMask(baseImg.size(), CV_WHITE_BYTE);

  ceres::Grid2D<unsigned char, 1> imgGrid(trackedImg.data, 0, trackedImg.rows,
                                          0, trackedImg.cols);
  ceres::BiCubicInterpolator<ceres::Grid2D<unsigned char, 1>> trackedFrame(
      imgGrid);

  ceres::Problem problem;

  problem.AddParameterBlock(motion.so3().data(), 4,
                            new ceres::EigenQuaternionParameterization());
  problem.AddParameterBlock(motion.translation().data(), 3);

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

  std::map<const ceres::CostFunction *, const PointTrackingResidual *>
      costFuncToResidual;
  StdVector<Vec2> gotOutside;

  int pntTotal = 0;

  for (const auto &point : depthLayer) {
    ++pntTotal;
    cv::Point cvp = toCvPoint(point.p);
    Vec3 pos = cam.unmap(point.p).normalized() * point.depth;
    if (!isPointTrackable(cam, pos, coarseMotion)) {
      gotOutside.push_back(point.p);
      continue;
    }

    ceres::LossFunction *lossFunc = nullptr;
    if (settings.frameTracker.useGradWeighting) {
      double gradNorm = gradNormAt(baseImg, cvp);
      double c = settings.gradWeighting.c;
      double weight = c / std::hypot(c, gradNorm);
      lossFunc = new ceres::ScaledLoss(
          new ceres::HuberLoss(settings.intencity.outlierDiff), weight,
          ceres::Ownership::TAKE_OWNERSHIP);
    } else
      lossFunc = new ceres::HuberLoss(settings.intencity.outlierDiff);

    auto newResidual = new PointTrackingResidual(
        pos, static_cast<double>(baseImg(cvp)), &cam, &trackedFrame);
    ceres::CostFunction *newCostFunc =
        new ceres::AutoDiffCostFunction<PointTrackingResidual, 1, 4, 3, 2>(
            newResidual);
    costFuncToResidual[newCostFunc] = newResidual;
    problem.AddResidualBlock(newCostFunc, lossFunc, motion.so3().data(),
                             motion.translation().data(), affLight.data);
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

  for (FrameTrackerObserver *obs : observers)
    obs->levelTracked(pyrLevel, motion, affLight, problem, summary,
                      costFuncToResidual);

  return {motion, affLight};
}

} // namespace fishdso
