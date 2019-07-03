#include "output/TrackingDebugImageDrawer.h"

namespace fishdso {

DEFINE_double(tracking_rel_point_size, 0.004,
              "Relative to w+h point size on tracking residuals images.");
DEFINE_int32(tracking_res_image_width, 1200,
             "Width of the image with tracking residuals.");
DEFINE_double(debug_max_residual, 12.0,
              "Max tracking residual when displaying debug image.");

TrackingDebugImageDrawer::TrackingDebugImageDrawer(
    const StdVector<CameraModel> &camPyr,
    const Settings::FrameTracker &frameTrackerSettings,
    const Settings::Pyramid &pyrSettings)
    : frameTrackerSettings(frameTrackerSettings)
    , pyrSettings(pyrSettings)
    , camPyr(camPyr)
    , residualsImg(
          pyrSettings.levelNum,
          cv::Mat3b::zeros(camPyr[0].getHeight(), camPyr[0].getWidth())) {}

void TrackingDebugImageDrawer::startTracking(const ImagePyramid &frame) {
  curFramePyr = frame.images;
  residualsImg.resize(0);
  residualsImg.resize(pyrSettings.levelNum);
}

void TrackingDebugImageDrawer::levelTracked(
    int levelNum, const SE3 &baseToLast,
    const AffineLightTransform<double> &affLightBaseToLast,
    const ceres::Problem &problem, const ceres::Solver::Summary &summary,
    const std::map<const ceres::CostFunction *, const PointTrackingResidual *>
        &costFuncToResidual) {
  int w = camPyr[levelNum].getWidth(), h = camPyr[levelNum].getHeight();
  int s = FLAGS_tracking_rel_point_size * (w + h) / 2;
  cv::cvtColor(curFramePyr[levelNum], residualsImg[levelNum],
               cv::COLOR_GRAY2BGR);

  std::vector<ceres::ResidualBlockId> residuals;
  problem.GetResidualBlocks(&residuals);
  for (auto resId : residuals) {
    auto costFunc = problem.GetCostFunctionForResidualBlock(resId);
    auto iter = costFuncToResidual.find(costFunc);
    if (iter == costFuncToResidual.end())
      continue;
    const PointTrackingResidual *residual = iter->second;
    const ceres::LossFunction *lossFunc =
        problem.GetLossFunctionForResidualBlock(resId);
    double eval = -1;
    double robustified[3] = {INF, 0, 0};
    (*residual)(baseToLast.unit_quaternion().coeffs().data(),
                baseToLast.translation().data(), affLightBaseToLast.data,
                &eval);
    lossFunc->Evaluate(eval * eval, robustified);
    robustified[0] = std::sqrt(robustified[0]);
    Vec2 onTracked = camPyr[levelNum].map(baseToLast * residual->pos);
    if (camPyr[levelNum].isOnImage(onTracked, 0))
      putSquare(residualsImg[levelNum], toCvPoint(onTracked), s,
                depthCol(robustified[0], 0, FLAGS_debug_max_residual),
                cv::FILLED);
  }
}

cv::Mat3b TrackingDebugImageDrawer::drawAllLevels() {
  return drawLeveled(residualsImg.data(), residualsImg.size(),
                     camPyr[0].getWidth(), camPyr[0].getHeight(),
                     FLAGS_tracking_res_image_width);
}

cv::Mat3b TrackingDebugImageDrawer::drawFinestLevel() {
  return residualsImg[0];
}

} // namespace fishdso
