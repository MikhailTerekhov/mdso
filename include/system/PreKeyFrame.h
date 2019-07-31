#ifndef INCLUDE_PREKEYFRAME
#define INCLUDE_PREKEYFRAME

#include "system/AffineLightTransform.h"
#include "system/CameraModel.h"
#include "util/ImagePyramid.h"
#include "util/settings.h"
#include "util/types.h"
#include <opencv2/core.hpp>
#include <sophus/se3.hpp>

namespace fishdso {

struct KeyFrame;
class PreKeyFrameInternals;

struct PreKeyFrame {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  PreKeyFrame(KeyFrame *baseKeyFrame, CameraModel *cam,
              const cv::Mat &frameColored, int globalFrameNum,
              const Settings::Pyramid &_pyrSettings = {});
  ~PreKeyFrame();

  cv::Mat frameColored;
  cv::Mat1d gradX, gradY, gradNorm;
  ImagePyramid framePyr;
  EIGEN_STRONG_INLINE cv::Mat1b &frame() { return framePyr[0]; }
  EIGEN_STRONG_INLINE const cv::Mat1b &frame() const { return framePyr[0]; }

  KeyFrame *baseKeyFrame;
  CameraModel *cam;
  SE3 baseToThis;
  AffineLightTransform<double> lightBaseToThis;
  int globalFrameNum;

  Settings::Pyramid pyrSettings;

  std::unique_ptr<PreKeyFrameInternals> internals;
};

} // namespace fishdso

#endif
