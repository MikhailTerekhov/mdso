#ifndef INCLUDE_IMAGEPYRAMID
#define INCLUDE_IMAGEPYRAMID

#include "util/settings.h"
#include <ceres/cubic_interpolation.h>
#include <opencv2/opencv.hpp>

namespace fishdso {

struct ImagePyramid {
  ImagePyramid(const cv::Mat1b &baseImage, int levelNum);

  inline cv::Mat1b &operator[](int ind) { return images[ind]; }
  inline const cv::Mat1b &operator[](int ind) const { return images[ind]; }

  std::vector<cv::Mat1b> images;
};

} // namespace fishdso

#endif
