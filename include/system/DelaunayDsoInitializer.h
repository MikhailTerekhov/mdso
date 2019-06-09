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
      const Settings::DelaunayDsoInitializer &initSettings = {},
      const Settings::StereoMatcher &smSettings = {},
      const Settings::Threading &threadingSettings = {},
      const Settings::Triangulation &triangulationSettings = {},
      const Settings::KeyFrame &kfSettings = {},
      const Settings::PointTracer &tracingSettings = {},
      const Settings::Intencity &intencitySettings = {},
      const Settings::ResidualPattern &rpSettings = {},
      const Settings::Pyramid &pyrSettings = {});

  // returns true if initialization is completed
  bool addFrame(const cv::Mat &frame, int globalFrameNum);

  std::vector<KeyFrame> createKeyFrames();

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

  Settings::DelaunayDsoInitializer initSettings;
  Settings::StereoMatcher smSettigns;
  Settings::Threading threadingSettings;
  Settings::Triangulation triangulationSettings;
  Settings::KeyFrame kfSettings;

  // TODO create PointTracer!!!
  Settings::PointTracer tracingSettings;
  Settings::Intencity intencitySettings;
  Settings::ResidualPattern rpSettings;
  Settings::Pyramid pyrSettings;

  std::vector<InitializerObserver *> observers;
};

} // namespace fishdso

#endif
