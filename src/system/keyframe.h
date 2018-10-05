#pragma once

#include "system/interestpoint.h"
#include "system/prekeyframe.h"
#include "util/settings.h"
#include <Eigen/StdVector>
#include <memory>
#include <opencv2/core.hpp>
#include <vector>

namespace fishdso {

struct KeyFrame {
  static constexpr int LI = settingInterestPointLayers;
  static constexpr int PL = settingPyrLevels;
  static int adaptiveBlockSize;

  KeyFrame(const cv::Mat &frameColored, int globalFrameNum);

  void setDepthPyrs();

  void selectPointsDenser(int pointsNeeded);

  std::unique_ptr<PreKeyFrame> preKeyFrame;
  cv::Mat frameColored;
  cv::Mat gradX, gradY, gradNorm;
  stdvectorInterestPoint interestPoints;

private:
  static void updateAdaptiveBlockSize(int pointsFound);

  int selectPoints(int blockSize, int pointsNeeded);

  int lastBlockSize;
  int lastPointsFound;
  int lastPointsUsed;
  bool areDepthsSet;
};

} // namespace fishdso
