#include "output/DepthPyramidDrawer.h"
#include <opencv2/opencv.hpp>

DEFINE_double(pyr_rel_point_size, 0.004,
              "Relative to w+h point size on the debug image pyramid.");
DEFINE_int32(pyr_image_width, 1200, "Width of the debug image pyramid.");

namespace mdso {

cv::Mat3b draw(const FrameTracker::DepthedMultiFrame &pyr) {
  std::vector<std::vector<cv::Mat3b>> images(pyr[0].images.size());
  std::vector<cv::Mat3b> levelsDrawn(pyr[0].images.size());
  for (int lvl = 0; lvl < pyr[0].images.size(); ++lvl) {
    images[lvl].resize(pyr.size());
    for (int camInd = 0; camInd < pyr.size(); ++camInd) {
      const cv::Mat1b &img = pyr[camInd].images[lvl];
      int s = FLAGS_pyr_rel_point_size * (img.cols + img.rows) / 2;
      images[lvl][camInd] = cvtGrayToBgr(img);
      for (const auto &[p, d] : pyr[camInd].depths[lvl])
        putSquare(images[lvl][camInd], toCvPoint(p), s,
                  depthCol(d, minDepthCol, maxDepthCol), cv::FILLED);
    }
    levelsDrawn[lvl] =
        drawLeveled(images[lvl].data(), images[lvl].size(), images[lvl][0].cols,
                    images[lvl][0].rows, images[lvl][0].cols);
  }
  return drawLeveled(levelsDrawn.data(), levelsDrawn.size(),
                     levelsDrawn[0].cols, levelsDrawn[0].rows,
                     FLAGS_pyr_image_width);
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
