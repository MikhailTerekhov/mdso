#include "output/DepthPyramidDrawer.h"
#include <opencv2/opencv.hpp>

DEFINE_double(pyr_rel_point_size, 0.004,
              "Relative to w+h point size on the debug image pyramid.");
DEFINE_int32(pyr_image_width, 1200, "Width of the debug image pyramid.");

namespace fishdso {

cv::Mat draw(const DepthedImagePyramid &pyr) {
  std::vector<cv::Mat3b> images(pyr.levelNum);
  for (int i = 0; i < pyr.levelNum; ++i) {
    int s = FLAGS_pyr_rel_point_size * (pyr[i].cols + pyr[i].rows) / 2;
    images[i] = cvtGrayToBgr(pyr[i]);
    for (const auto &p : pyr.depthPyr[i])
      putSquare(images[i], toCvPoint(p.p), s,
                depthCol(p.depth, minDepthCol, maxDepthCol), cv::FILLED);
  }
  return drawLeveled(images.data(), pyr.levelNum, pyr[0].cols, pyr[0].rows,
                     FLAGS_pyr_image_width);
}

void DepthPyramidDrawer::newBaseFrame(const DepthedImagePyramid &pyr) {
  mPyrChanged = true;
  lastPyr = draw(pyr);
}

bool DepthPyramidDrawer::pyrChanged() { return mPyrChanged; }

cv::Mat DepthPyramidDrawer::getLastPyr() {
  mPyrChanged = false;
  return lastPyr;
}

} // namespace fishdso
