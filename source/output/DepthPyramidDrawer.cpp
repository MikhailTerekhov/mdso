#include "output/DepthPyramidDrawer.h"
#include <opencv2/opencv.hpp>

DEFINE_double(pyr_rel_point_size, 0.004,
              "Relative to w+h point size on the debug image pyramid.");
DEFINE_int32(pyr_image_width, 1200, "Width of the debug image pyramid.");

namespace mdso {

cv::Mat3b draw(const FrameTracker::DepthedMultiFrame &pyr) {
  std::vector<cv::Mat3b> levelsDrawn(pyr[0].images.size());
  for (int lvl = 0; lvl < pyr[0].images.size(); ++lvl) {
    std::vector<cv::Mat3b> curImages(pyr.size());
    for (int camInd = 0; camInd < pyr.size(); ++camInd) {
      const cv::Mat1b &img = pyr[camInd].images[lvl];
      cv::Mat3b drawnUnresized = cvtGrayToBgr(img);
      int s = FLAGS_pyr_rel_point_size * (img.cols + img.rows) / 2;
      for (const auto &[p, d] : pyr[camInd].depths[lvl])
        putSquare(drawnUnresized, toCvPoint(p), s,
                  depthCol(d, minDepthCol, maxDepthCol), cv::FILLED);
      cv::resize(drawnUnresized, curImages[camInd],
                 pyr[camInd].images[0].size(), 0, 0, cv::INTER_NEAREST);
    }
    cv::vconcat(curImages.data(), curImages.size(), levelsDrawn[lvl]);
  }

  cv::Mat3b result;
  cv::hconcat(levelsDrawn.data(), levelsDrawn.size(), result);
  return result;
}

void DepthPyramidDrawer::newBaseFrame(
    const FrameTracker::DepthedMultiFrame &pyr) {
  mPyrChanged = true;

  lastPyr = draw(pyr);
}

bool DepthPyramidDrawer::pyrChanged() { return mPyrChanged; }

cv::Mat3b DepthPyramidDrawer::getLastPyr() {
  mPyrChanged = false;
  return lastPyr;
}

} // namespace mdso
