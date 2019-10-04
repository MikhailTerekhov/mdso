#include "output/TrackingDebugImageDrawer.h"

namespace mdso {

DEFINE_double(tracking_rel_point_size, 0.004,
              "Relative to w+h point size on tracking residuals images.");
DEFINE_int32(tracking_res_image_width, 1200,
             "Width of the image with tracking residuals.");
DEFINE_double(debug_max_residual, 12.0,
              "Max tracking residual when displaying debug image.");

TrackingDebugImageDrawer::TrackingDebugImageDrawer(
    CameraBundle camPyr[], const Settings::FrameTracker &frameTrackerSettings,
    const Settings::Pyramid &pyrSettings, const std::vector<int> &drawingOrder)
    : frameTrackerSettings(frameTrackerSettings)
    , pyrSettings(pyrSettings)
    , camPyr(camPyr)
    , curFramePyr(pyrSettings.levelNum())
    , residualsImg(pyrSettings.levelNum())
    , drawingOrder(drawingOrder) {
  for (auto &r : residualsImg)
    r.resize(camPyr[0].bundle.size());
  for (auto &f : curFramePyr)
    f.resize(camPyr[0].bundle.size());
}

void TrackingDebugImageDrawer::startTracking(const PreKeyFrame &frame) {
  for (int camInd = 0; camInd < frame.frames.size(); ++camInd)
    for (int lvl = 0; lvl < pyrSettings.levelNum(); ++lvl)
      curFramePyr[lvl][camInd] = frame.frames[camInd].framePyr[lvl];
}

void TrackingDebugImageDrawer::levelTracked(
    int pyrLevel, const TrackingResult &result,
    const std::vector<StdVector<std::pair<Vec2, double>>> &pointResiduals) {
  CHECK(pointResiduals.size() == curFramePyr[0].size());
  for (int camInd = 0; camInd < pointResiduals.size(); ++camInd) {
    int w = camPyr[pyrLevel].bundle[camInd].cam.getWidth(),
        h = camPyr[pyrLevel].bundle[camInd].cam.getHeight();
    int s = FLAGS_tracking_rel_point_size * (w + h) / 2;

    cv::cvtColor(curFramePyr[pyrLevel][camInd], residualsImg[pyrLevel][camInd],
                 cv::COLOR_GRAY2BGR);

    for (const auto &[point, res] : pointResiduals[camInd]) {
      if (camPyr[pyrLevel].bundle[camInd].cam.isOnImage(point, s)) {
        putSquare(residualsImg[pyrLevel][camInd], toCvPoint(point), s,
                  depthCol(std::abs(res), 0, FLAGS_debug_max_residual),
                  cv::FILLED);
      }
    }
  }
}

cv::Mat3b TrackingDebugImageDrawer::drawAllLevels() {
  std::vector<cv::Mat3b> levels(pyrSettings.levelNum());
  for (int lvl = 0; lvl < levels.size(); ++lvl) {
    std::vector<cv::Mat3b> resized(residualsImg[lvl].size());
    for (int camInd = 0; camInd < resized.size(); ++camInd)
      cv::resize(residualsImg[lvl][camInd], resized[camInd],
                 residualsImg[0][0].size(), 0, 0, cv::INTER_NEAREST);
    cv::vconcat(resized.data(), residualsImg[lvl].size(), levels[lvl]);
  }
  cv::Mat3b result;
  cv::hconcat(levels.data(), levels.size(), result);
  return result;
}

cv::Mat3b TrackingDebugImageDrawer::drawFinestLevel() { return drawLevel(0); }

cv::Mat3b TrackingDebugImageDrawer::drawLevel(int pyrLevel) {
  CHECK(pyrLevel >= 0 && pyrLevel < pyrSettings.levelNum());
  cv::Mat3b result = drawLeveled(
      residualsImg[pyrLevel].data(), residualsImg[pyrLevel].size(),
      camPyr[0].bundle[0].cam.getWidth(), camPyr[0].bundle[0].cam.getHeight(),
      camPyr[0].bundle[0].cam.getWidth());
  return result;
}

} // namespace mdso
