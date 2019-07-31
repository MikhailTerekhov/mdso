#ifndef INCLUDE_DELAUNAYDSOINITIALIZER
#define INCLUDE_DELAUNAYDSOINITIALIZER

#include "DsoInitializer.h"
#include "DsoSystem.h"
#include "output/InitializerObserver.h"
#include "system/KeyFrame.h"
#include "system/StereoMatcher.h"
#include <memory>
#include <opencv2/opencv.hpp>

namespace fishdso {

class DelaunayDsoInitializer : public DsoInitializer {
public:
  enum DebugOutputType { NO_DEBUG, SPARSE_DEPTHS, FILLED_DEPTHS };

  DelaunayDsoInitializer(
      DsoSystem *dsoSystem, CameraModel *cam, PixelSelector *pixelSelector,
      int pointsNeeded, DebugOutputType debugOutputType,
      const std::vector<InitializerObserver *> &observers = {},
      const InitializerSettings &settings = {});

  // returns true if initialization is completed
  bool addFrame(const cv::Mat &frame, int globalFrameNum);

  StdVector<KeyFrame> createKeyFrames();

private:
  CameraModel *cam;
  DsoSystem *dsoSystem;
  PixelSelector *pixelSelector;
  StereoMatcher stereoMatcher;
  bool hasFirstFrame;
  int framesSkipped;
  cv::Mat frames[2];
  int globalFrameNums[2];
  int pointsNeeded;
  DebugOutputType debugOutputType;
  InitializerSettings settings;
  std::vector<InitializerObserver *> observers;
};

} // namespace fishdso

#endif
