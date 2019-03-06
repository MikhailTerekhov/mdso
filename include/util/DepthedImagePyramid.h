#ifndef INCLUDE_DEPTHEDIMAGEPYRAMID
#define INCLUDE_DEPTHEDIMAGEPYRAMID

#include "util/ImagePyramid.h"
#include "util/settings.h"
#include <opencv2/opencv.hpp>

namespace fishdso {

struct DepthedImagePyramid : ImagePyramid {
  DepthedImagePyramid(const cv::Mat1b &baseImage, const StdVector<Vec2> &points,
                      const std::vector<double> &depthsVec,
                      const std::vector<double> &weightsVec);

  cv::Mat3b draw();

  std::array<cv::Mat1d, settingPyrLevels> depths;
};

} // namespace fishdso

#endif
