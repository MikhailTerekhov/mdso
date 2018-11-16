#ifndef INCLUDE_IMAGEPYRAMID
#define INCLUDE_IMAGEPYRAMID

#include "util/settings.h"
#include <opencv2/opencv.hpp>

namespace fishdso {

struct ImagePyramid {
  ImagePyramid(const cv::Mat1b &baseImage);

  std::array<cv::Mat1b, settingPyrLevels> images;
};

} // namespace fishdso

#endif
