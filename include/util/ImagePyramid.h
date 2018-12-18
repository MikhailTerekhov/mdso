#ifndef INCLUDE_IMAGEPYRAMID
#define INCLUDE_IMAGEPYRAMID

#include "util/settings.h"
#include <opencv2/opencv.hpp>

namespace fishdso {

struct ImagePyramid {
  ImagePyramid(const cv::Mat1b &baseImage);

  inline cv::Mat1b &operator[](int ind) { return images[ind]; }
  inline const cv::Mat1b &operator[](int ind) const { return images[ind]; }

  std::array<cv::Mat1b, settingPyrLevels> images;
};

} // namespace fishdso

#endif
