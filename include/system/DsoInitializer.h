#ifndef INCLUDE_DSOINITIALIZER
#define INCLUDE_DSOINITIALIZER

#include "system/KeyFrame.h"
#include "system/StereoMatcher.h"
#include <memory>
#include <opencv2/opencv.hpp>

namespace fishdso {

class CameraModel;

class DsoInitializer {
public:
  virtual ~DsoInitializer() {}

  // returns true if initialization is completed
  virtual bool addFrame(const cv::Mat &frame, int globalFrameNum) = 0;

  virtual std::vector<KeyFrame> createKeyFrames() = 0;
};

} // namespace fishdso

#endif
