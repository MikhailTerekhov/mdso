#pragma once

#include "system/interestpoint.h"
#include "util/settings.h"
#include <memory>
#include <opencv2/core.hpp>
#include <vector>

namespace fishdso {

struct KeyFrame {
  static constexpr int L = settingInterestPointLayers;
  static int adaptiveBlockSize;

  KeyFrame(const cv::Mat &frameColored);

  cv::Mat frame;
  cv::Mat frameColored;
  cv::Mat gradX, gradY, gradNorm;
  std::vector<InterestPoint> interestPoints;

#ifdef DEBUG
  cv::Mat frameWithPoints;
#endif

private:
  static void updateAdaptiveBlockSize(int pointsFound);

  void selectPoints();
};

} // namespace fishdso
