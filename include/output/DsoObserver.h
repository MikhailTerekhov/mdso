#ifndef INCLUDE_DSOOBSERVER
#define INCLUDE_DSOOBSERVER

#include "system/CameraModel.h"
#include "system/KeyFrame.h"
#include "system/PreKeyFrame.h"
#include <opencv2/opencv.hpp>

namespace fishdso {

class DsoSystem;

class DsoObserver {
public:
  virtual ~DsoObserver() = 0;

  virtual void created(DsoSystem *newDso, CameraBundle *newCam,
                       const Settings &newSettings) {}
  virtual void initialized(const KeyFrame *initializedKFs[], int size) {}
  virtual void newFrame(const PreKeyFrame &frame) {}
  virtual void newBaseFrame(const KeyFrame &baseFrame) {}
  virtual void keyFramesMarginalized(const KeyFrame *marginalized[], int size) {
  }
  virtual void destructed(const KeyFrame *lastKeyFrames[], int size) {}
};

} // namespace fishdso

#endif
