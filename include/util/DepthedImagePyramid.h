#ifndef INCLUDE_DEPTHEDIMAGEPYRAMID
#define INCLUDE_DEPTHEDIMAGEPYRAMID

#include "util/ImagePyramid.h"
#include "util/settings.h"
#include <opencv2/opencv.hpp>

namespace mdso {

struct DepthedImagePyramid : ImagePyramid {
  struct Point {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Vec2 p;
    double depth;
  };

  DepthedImagePyramid(const cv::Mat1b &baseImage, int levelNum, Vec2 points[],
                      double depthsArray[], double weightsArray[], int size);

  std::vector<StdVector<Point>> depths;
};

} // namespace mdso

#endif
