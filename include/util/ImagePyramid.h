#ifndef INCLUDE_IMAGEPYRAMID
#define INCLUDE_IMAGEPYRAMID

#include "util/settings.h"
#include <ceres/cubic_interpolation.h>
#include <opencv2/opencv.hpp>

namespace fishdso {

struct ImagePyramid {
  ImagePyramid(const cv::Mat1b &baseImage, int levelNum);

  inline cv::Mat1b &operator[](int ind) {
    CHECK(ind >= 0 && ind < levelNum);
    return images[ind];
  }
  inline const cv::Mat1b &operator[](int ind) const {
    CHECK(ind >= 0 && ind < levelNum);
    return images[ind];
  }
  inline ceres::BiCubicInterpolator<ceres::Grid2D<unsigned char, 1>> &
  interpolator(int ind) {
    CHECK(ind >= 0 && ind < levelNum);
    return *interpolators[ind];
  }
  inline const ceres::BiCubicInterpolator<ceres::Grid2D<unsigned char, 1>> &
  interpolator(int ind) const {
    CHECK(ind >= 0 && ind < levelNum);
    return *interpolators[ind];
  }

  int levelNum;
  cv::Mat1b images[Settings::Pyramid::max_levelNum];
  std::unique_ptr<ceres::Grid2D<unsigned char, 1>>
      grids[Settings::Pyramid::max_levelNum];
  std::unique_ptr<ceres::BiCubicInterpolator<ceres::Grid2D<unsigned char, 1>>>
      interpolators[Settings::Pyramid::max_levelNum];
};

} // namespace fishdso

#endif
