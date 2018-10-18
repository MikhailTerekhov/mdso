#ifndef INCLUDE_PREKEYFRAME
#define INCLUDE_PREKEYFRAME

#include "system/AffineLightTransform.h"
#include "util/settings.h"
#include "util/types.h"
#include <opencv2/core.hpp>
#include <sophus/se3.hpp>

namespace fishdso {

struct PreKeyFrame {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  static constexpr int PL = settingPyrLevels;

  PreKeyFrame(const cv::Mat &frameColored, int globalFrameNum);

  cv::Mat1b &frame();
  void setDepthPyrs(const cv::Mat1d &depths0, const cv::Mat1d &weights);

  cv::Mat drawDepthedFrame(int pyrLevel, double minDepth, double maxDepth);

  cv::Mat1b framePyr[PL];
  cv::Mat1d depths[PL];
  SE3 worldToThis;
  AffineLightTransform<double> lightWorldToThis;
  int globalFrameNum;
  bool areDepthsSet;
};

} // namespace fishdso

#endif
