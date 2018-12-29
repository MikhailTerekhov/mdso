#ifndef INCLUDE_IMAGEPYRAMID
#define INCLUDE_IMAGEPYRAMID

#include "util/settings.h"
#include <ceres/cubic_interpolation.h>
#include <opencv2/opencv.hpp>

namespace fishdso {

struct ImagePyramid {
  ImagePyramid(const cv::Mat1b &baseImage);

  inline cv::Mat1b &operator[](int ind) { return images[ind]; }
  inline const cv::Mat1b &operator[](int ind) const { return images[ind]; }
  inline ceres::BiCubicInterpolator<ceres::Grid2D<unsigned char, 1>> &
  interpolator(int ind) {
    return *interpolators[ind];
  }
  inline const ceres::BiCubicInterpolator<ceres::Grid2D<unsigned char, 1>> &
  interpolator(int ind) const {
    return *interpolators[ind];
  }

  std::array<cv::Mat1b, settingPyrLevels> images;
  std::array<std::unique_ptr<ceres::Grid2D<unsigned char, 1>>, settingPyrLevels>
      grids;
  std::array<std::unique_ptr<
                 ceres::BiCubicInterpolator<ceres::Grid2D<unsigned char, 1>>>,
             settingPyrLevels>
      interpolators;
};

} // namespace fishdso

#endif
