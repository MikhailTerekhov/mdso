#pragma once

#include "system/cameramodel.h"
#include "system/dsoinitializer.h"
#include "system/keyframe.h"
#include <map>
#include <memory>
#include <opencv2/core.hpp>

namespace fishdso {

struct KeyFrame;

class DsoSystem {
public:
  DsoSystem(CameraModel *cam);

  void addFrame(const cv::Mat &frame);

private:
  void addKf(cv::Mat frameColored);

  CameraModel *cam;

  DsoInitializer dsoInitializer;
  bool isInitialized;
};

} // namespace fishdso
