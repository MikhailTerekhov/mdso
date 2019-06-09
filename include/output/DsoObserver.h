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
  virtual void created(DsoSystem *newDso, CameraModel *newCam,
                       const Settings &newSettings) {}
  virtual void initialized(const std::vector<const KeyFrame *> &initializedKFs) {}
  virtual void newFrame(const PreKeyFrame *frame) {}
  virtual void newKeyFrame(const KeyFrame *baseFrame) {}
  virtual void
  keyFramesMarginalized(const std::vector<const KeyFrame *> &marginalized) {}
  virtual void destructed(const std::vector<const KeyFrame *> &lastKeyFrames) {}

protected:
  DsoObserver() {}
};

} // namespace fishdso

#endif
