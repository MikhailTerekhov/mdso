#ifndef INCLUDE_DEPTHEDIMAGEPYRAMID
#define INCLUDE_DEPTHEDIMAGEPYRAMID

#include "util/ImagePyramid.h"
#include "util/settings.h"
#include <opencv2/opencv.hpp>

namespace mdso {

struct DepthedImagePyramid : ImagePyramid {
  DepthedImagePyramid(const cv::Mat1b &baseImage, int levelNum, Vec2 points[],
                      double depthsArray[], double weightsArray[], int size);

  static_vector<cv::Mat1d, Settings::Pyramid::max_levelNum> depths;
};

} // namespace mdso

#endif
