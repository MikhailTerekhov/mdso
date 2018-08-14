#pragma once

#include "cameramodel.h"
#include "keyframe.h"
#include <map>
#include <memory>
#include <opencv2/core.hpp>

namespace fishdso {

class KeyFrame;

class DsoSystem {
  friend class KeyFrame;

public:
  DsoSystem(const CameraModel &cam);

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

  CameraModel cam;
};

} // namespace fishdso
