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

  PreKeyFrame(CameraModel *cam, const cv::Mat &frameColored, int globalFrameNum,
              const Settings::Pyramid &_pyrSettings = {});

  cv::Mat frameColored;
  cv::Mat1d gradX, gradY, gradNorm;
  ImagePyramid framePyr;
  EIGEN_STRONG_INLINE cv::Mat1b &frame() { return framePyr[0]; }
  EIGEN_STRONG_INLINE const cv::Mat1b &frame() const { return framePyr[0]; }

  CameraModel *cam;
  SE3 worldToThis;
  AffineLightTransform<double> lightWorldToThis;
  int globalFrameNum;

  Settings::Pyramid pyrSettings;
};

} // namespace fishdso

#endif
