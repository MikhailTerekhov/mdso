#pragma once

#include "system/cameramodel.h"
#include "system/dsoinitializer.h"
#include "system/keyframe.h"
#include <map>
#include <memory>
#include <opencv2/core.hpp>

namespace fishdso {

class DsoInitializer;
struct KeyFrame;

class DsoSystem {
  friend class DsoInitializer;
  friend struct KeyFrame;

public:
  DsoSystem(CameraModel *cam);

  void addFrame(const cv::Mat &frame);
  void addKf(cv::Mat frameColored);
  void removeKf();

  void updateAdaptiveBlockSize(int curPointsDetected);

  void showDebug() const;

private:
  std::map<int, std::unique_ptr<KeyFrame>> keyframes;
  int curFrameId;
  int curPointId;

  // for keyframe point-of-interest detection
  int adaptiveBlockSize;

  CameraModel *cam;

  std::unique_ptr<DsoInitializer> dsoInitializer;
  bool isInitialized;
};

} // namespace fishdso
