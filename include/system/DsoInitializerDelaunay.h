#ifndef INCLUDE_DELAUNAYDSOINITIALIZER
#define INCLUDE_DELAUNAYDSOINITIALIZER

#include "output/DelaunayInitializerObserver.h"
#include "system/DsoInitializer.h"
#include "system/DsoSystem.h"
#include "system/KeyFrame.h"
#include "system/StereoMatcher.h"
#include <memory>
#include <opencv2/opencv.hpp>

namespace mdso {

class DsoInitializerDelaunay : public DsoInitializer {
public:
  DsoInitializerDelaunay(
      CameraBundle *cam, PixelSelector pixelSelectors[],
      const std::vector<DelaunayInitializerObserver *> &observers = {},
      const InitializerDelaunaySettings &settings = {});

  bool addMultiFrame(const cv::Mat newFrames[],
                     Timestamp timestamps[]) override;

  InitializedVector initialize() override;

private:
  void setImages();

  CameraBundle *cam;
  PixelSelector *pixelSelectors;
  bool hasFirstFrame;
  int framesSkipped;
  cv::Mat frames[2];
  cv::Mat1b framesGray[2];
  cv::Mat1d gradNorm[2];
  Timestamp timestamps[2];
  InitializerDelaunaySettings settings;
  std::vector<DelaunayInitializerObserver *> observers;
};

} // namespace mdso

#endif
