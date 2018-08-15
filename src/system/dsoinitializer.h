#pragma once
#include "dsosystem.h"
#include "stereomatcher.h"
#include <memory>
#include <opencv2/opencv.hpp>

namespace fishdso {

class DsoSystem;
class CameraModel;

class DsoInitializer {
public:
  DsoInitializer(DsoSystem *dsoSystem);

  // returns true if initialization is completed
  bool addFrame(const cv::Mat &frame);

private:
  void addFirstFrame(const cv::Mat &frame);

  DsoSystem *dsoSystem;
  std::unique_ptr<StereoMatcher> stereoMatcher;
  bool hasFirstFrame;
  int framesSkipped;
};

} // namespace fishdso
