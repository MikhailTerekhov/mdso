#ifndef INCLUDE_DELAUNAYDSOINITIALIZER
#define INCLUDE_DELAUNAYDSOINITIALIZER

#include "DsoInitializer.h"
#include "DsoSystem.h"
#include "output/DelaunayInitializerObserver.h"
#include "system/KeyFrame.h"
#include "system/StereoMatcher.h"
#include <memory>
#include <opencv2/opencv.hpp>

namespace fishdso {

class DelaunayDsoInitializer : public DsoInitializer {
public:
  // DelaunayDsoInitializer(
  // DsoSystem *dsoSystem, CameraBundle *cam, PixelSelector pixelSelectors[],
  // int pointsNeeded,
  // const std::vector<InitializerObserver *> &observers = {},
  // const InitializerSettings &settings = {});

  // returns true if initialization is completed
  bool addMultiFrame(const cv::Mat frame[]);

  InitializedVector initialize();

  // private:
  // CameraBundle *cam;
  // DsoSystem *dsoSystem;
  // PixelSelector *pixelSelectors;
  // bool hasFirstFrame;
  // int framesSkipped;
  // cv::Mat frames[2];
  // int globalFrameNums[2];
  // int pointsNeeded;
  // InitializerSettings settings;
  // std::vector<InitializerObserver *> observers;
};

} // namespace fishdso

#endif
