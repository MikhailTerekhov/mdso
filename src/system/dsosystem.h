#pragma once

#include "keyframe.h"
#include <map>
#include <memory>
#include <opencv2/core.hpp>

namespace fishdso {

class KeyFrame;

class DsoSystem {
  friend class KeyFrame;

public:
  DsoSystem();

  void addKf(cv::Mat frameColored);
  void removeKf();

  void updateAdaptiveBlockSize(int curPointsDetected);

  void showDebug();

private:
  std::map<int, std::unique_ptr<KeyFrame>> keyframes;
  int curFrameId;
  int curPointId;

  // for keyframe point-of-interest detection
  int adaptiveBlockSize;
};

} // namespace fishdso
