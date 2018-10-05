#pragma once

#include "util/settings.h"
#include "util/types.h"
#include <opencv2/core.hpp>
#include <sophus/se3.hpp>

namespace fishdso {

struct PreKeyFrame {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  static constexpr int PL = settingPyrLevels;

  PreKeyFrame(const cv::Mat &frameColored, int globalFrameNum);

  void setDepthPyrs(const cv::Mat1d &depths0, const cv::Mat1d &weights);

  cv::Mat drawDepthedFrame(int pyrLevel, double minDepth, double maxDepth);

  cv::Mat1b framePyr[PL];
  cv::Mat1d depths[PL];
  SE3 worldToThis;
  int globalFrameNum;
  bool areDepthsSet;
};

} // namespace fishdso
