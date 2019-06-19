#ifndef INCLUDE_DEPTHEDIMAGEPYRAMID
#define INCLUDE_DEPTHEDIMAGEPYRAMID

#include "util/ImagePyramid.h"
#include "util/settings.h"
#include <opencv2/opencv.hpp>

namespace fishdso {

struct DepthedImagePyramid : ImagePyramid {

  struct Point {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Vec2 p;
    double depth;
    double weight;
  };

  DepthedImagePyramid(const cv::Mat1b &baseImage, int levelNum,
                      const StdVector<Point> &points);

  StdVector<Point> depthPyr[Settings::Pyramid::max_levelNum];
};

} // namespace fishdso

#endif
