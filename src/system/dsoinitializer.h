#pragma once
#include "system/keyframe.h"
#include "system/stereomatcher.h"
#include <memory>
#include <opencv2/opencv.hpp>

namespace fishdso {

class CameraModel;

class DsoInitializer {
public:
  enum InterpolationType { NORMAL, PLAIN };
  enum DebugOutputType { NO_DEBUG, SPARSE_DEPTHS, FILLED_DEPTHS };

  DsoInitializer(CameraModel *cam);

  // returns true if initialization is completed
  bool addFrame(const cv::Mat &frame);

  std::vector<KeyFrame> createKeyFrames(DebugOutputType debugOutputType);

private:
  void addFirstFrame(const cv::Mat &frame);
  std::vector<KeyFrame>
  createKeyFramesFromStereo(InterpolationType interpolationType,
                            DebugOutputType debugOutputType);

  CameraModel *cam;
  StereoMatcher stereoMatcher;
  bool hasFirstFrame;
  int framesSkipped;
  cv::Mat frames[2];
};

} // namespace fishdso
