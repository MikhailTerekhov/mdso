#pragma once

#include "system/interestpoint.h"
#include "util/settings.h"
#include <memory>
#include <opencv2/core.hpp>
#include <vector>

namespace fishdso {

struct KeyFrame {
  static constexpr int LI = settingInterestPointLayers;
  static constexpr int PL = settingPyrLevels;
  static int adaptiveBlockSize;

  KeyFrame(const cv::Mat &frameColored);
  
  void setDepthPyrs();

  cv::Mat1f depths[PL];

  cv::Mat framePyr[PL];
  cv::Mat frameColored;
  cv::Mat gradX, gradY, gradNorm;
  std::vector<InterestPoint> interestPoints;

  cv::Mat drawDepthedFrame(int pyrLevel, double minDepth, double maxDepth);

private:
  static void updateAdaptiveBlockSize(int pointsFound);

  void selectPoints();
  void setImgPyrs();

  bool areDepthsSet;
};

} // namespace fishdso
