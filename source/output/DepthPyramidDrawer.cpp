#include "output/DepthPyramidDrawer.h"
#include <opencv2/opencv.hpp>

DEFINE_double(pyr_rel_point_size, 0.004,
              "Relative to w+h point size on the debug image pyramid.");
DEFINE_int32(pyr_image_width, 1200, "Width of the debug image pyramid.");

namespace fishdso {

cv::Mat draw(const DepthedImagePyramid &pyr) {
  std::vector<cv::Mat3b> images(pyr.images.size());
  for (int i = 0; i < pyr.images.size(); ++i) {
    int s = FLAGS_pyr_rel_point_size * (pyr[i].cols + pyr[i].rows) / 2;
    images[i] = cvtGrayToBgr(pyr[i]);
    for (int y = 0; y < pyr.depths[i].rows; ++y)
      for (int x = 0; x < pyr.depths[i].cols; ++x)
        if (pyr.depths[i](y, x) > 0)
          putSquare(images[i], cv::Point(x, y), s,
                    depthCol(pyr.depths[i](y, x), minDepthCol, maxDepthCol),
                    cv::FILLED);
  }
  return drawLeveled(images.data(), pyr.depths.size(), pyr[0].cols, pyr[0].rows,
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
