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

  cv::Mat drawDepthedFrame(int pyrLevel, double minDepth, double maxDepth);

  cv::Mat1b frame;
  SE3 worldToThis;
  AffineLightTransform<double> lightWorldToThis;
  int globalFrameNum;
  bool areDepthsSet;
};

} // namespace fishdso

#endif
