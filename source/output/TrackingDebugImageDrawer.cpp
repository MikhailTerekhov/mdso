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
    const StdVector<std::pair<Vec2, double>> &pointResiduals) {
  int w = camPyr[levelNum].getWidth(), h = camPyr[levelNum].getHeight();
  int s = FLAGS_tracking_rel_point_size * (w + h) / 2;
  cv::cvtColor(curFramePyr[levelNum], residualsImg[levelNum],
               cv::COLOR_GRAY2BGR);

  for (const auto &[point, res] : pointResiduals)
    if (camPyr[levelNum].isOnImage(point, s))
      putSquare(residualsImg[levelNum], toCvPoint(point), s,
                depthCol(std::abs(res), 0, FLAGS_debug_max_residual), cv::FILLED);
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
