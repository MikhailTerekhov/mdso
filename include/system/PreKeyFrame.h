#ifndef INCLUDE_PREKEYFRAME
#define INCLUDE_PREKEYFRAME

#include "system/AffineLightTransform.h"
#include "system/CameraModel.h"
#include "util/ImagePyramid.h"
#include "util/settings.h"
#include "util/types.h"
#include <ceres/cubic_interpolation.h>
#include <opencv2/core.hpp>
#include <sophus/se3.hpp>

namespace fishdso {

struct PreKeyFrame {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  static constexpr int PL = settingPyrLevels;

  PreKeyFrame(CameraModel *cam, const cv::Mat &frameColored,
              int globalFrameNum);

  cv::Mat frameColored;
  ImagePyramid framePyr;
  EIGEN_STRONG_INLINE cv::Mat1b &frame() { return framePyr[0]; }
  EIGEN_STRONG_INLINE const cv::Mat1b &frame() const { return framePyr[0]; }

  ceres::Grid2D<unsigned char, 1> frameGrid;
  ceres::BiCubicInterpolator<ceres::Grid2D<unsigned char, 1>> frameInterpolator;
  CameraModel *cam;
  SE3 worldToThis;
  AffineLightTransform<double> lightWorldToThis;
  int globalFrameNum;
};

} // namespace fishdso

#endif
