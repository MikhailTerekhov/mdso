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
  DelaunayDsoInitializer(
      DsoSystem *dsoSystem, CameraBundle *cam, PixelSelector pixelSelectors[],
      const std::vector<DelaunayInitializerObserver *> &observers = {},
      const InitializerSettings &settings = {});

  bool addMultiFrame(const cv::Mat newFrames[],
                     Timestamp timestamps[]) override;

  InitializedVector initialize() override;

private:
  void setImages();

  CameraBundle *cam;
  DsoSystem *dsoSystem;
  PixelSelector *pixelSelectors;
  bool hasFirstFrame;
  int framesSkipped;
  cv::Mat frames[2];
  cv::Mat1b framesGray[2];
  cv::Mat1d gradNorm[2];
  Timestamp timestamps[2];
  InitializerSettings settings;
  std::vector<DelaunayInitializerObserver *> observers;
};

} // namespace fishdso

#endif
