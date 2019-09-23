#include "output/TrackingDebugImageDrawer.h"

namespace fishdso {

DEFINE_double(tracking_rel_point_size, 0.004,
              "Relative to w+h point size on tracking residuals images.");
DEFINE_int32(tracking_res_image_width, 1200,
             "Width of the image with tracking residuals.");
DEFINE_double(debug_max_residual, 12.0,
              "Max tracking residual when displaying debug image.");

TrackingDebugImageDrawer::TrackingDebugImageDrawer(
    CameraBundle camPyr[], const Settings::FrameTracker &frameTrackerSettings,
    const Settings::Pyramid &pyrSettings)
    : frameTrackerSettings(frameTrackerSettings)
    , pyrSettings(pyrSettings)
    , camPyr(camPyr)
    , residualsImg(pyrSettings.levelNum(),
                   cv::Mat3b::zeros(camPyr[0].bundle[0].cam.getHeight(),
                                    camPyr[0].bundle[0].cam.getWidth())) {
  curFramePyr.reserve(pyrSettings.levelNum());
}

void TrackingDebugImageDrawer::startTracking(const PreKeyFrame &frame) {
  curFramePyr.resize(0);
  for (const cv::Mat1b &img : frame.frames[0].framePyr.images)
    curFramePyr.push_back(img.clone());
  residualsImg.resize(0);
  residualsImg.resize(pyrSettings.levelNum());
}

void TrackingDebugImageDrawer::levelTracked(
    int pyrLevel, const FrameTracker::TrackingResult &result,
    std::pair<Vec2, double> pointResiduals[], int size) {
  CHECK(camPyr[0].bundle.size() == 1) << "Multicamera case is NIY";

  std::sort(pointResiduals, pointResiduals + size,
            [](const auto &a, const auto &b) {
              return a.first[0] == b.first[0] ? a.first[1] < b.first[1]
                                              : a.first[0] < b.first[0];
            });
  std::stringstream vlogPnt;
  vlogPnt << "Residual points: ";
  for (int i = 0; i < size; ++i)
    vlogPnt << "(" << pointResiduals[i].first.transpose() << ") ";
  VLOG(2) << vlogPnt.str();

  int w = camPyr[pyrLevel].bundle[0].cam.getWidth(),
      h = camPyr[pyrLevel].bundle[0].cam.getHeight();
  int s = FLAGS_tracking_rel_point_size * (w + h) / 2;

  cv::cvtColor(curFramePyr[pyrLevel], residualsImg[pyrLevel],
               cv::COLOR_GRAY2BGR);

  for (int i = 0; i < size; ++i) {
    const auto &[point, res] = pointResiduals[i];
    if (camPyr[pyrLevel].bundle[0].cam.isOnImage(point, s)) {
      putSquare(residualsImg[pyrLevel], toCvPoint(point), s,
                depthCol(std::abs(res), 0, FLAGS_debug_max_residual),
                cv::FILLED);
    }
  }
}

cv::Mat3b TrackingDebugImageDrawer::drawAllLevels() {
  return drawLeveled(residualsImg.data(), residualsImg.size(),
                     camPyr[0].bundle[0].cam.getWidth(),
                     camPyr[0].bundle[0].cam.getHeight(),
                     FLAGS_tracking_res_image_width);
}

cv::Mat3b TrackingDebugImageDrawer::drawFinestLevel() {
  return residualsImg[0];
}

} // namespace fishdso
