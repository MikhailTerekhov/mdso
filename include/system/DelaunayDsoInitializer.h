#ifndef INCLUDE_DELAUNAYDSOINITIALIZER
#define INCLUDE_DELAUNAYDSOINITIALIZER

#include "DsoInitializer.h"
#include "DsoSystem.h"
#include "system/KeyFrame.h"
#include "system/StereoMatcher.h"
#include <memory>
#include <opencv2/opencv.hpp>

namespace fishdso {

class DelaunayDsoInitializer : public DsoInitializer {
public:
  enum DebugOutputType { NO_DEBUG, SPARSE_DEPTHS, FILLED_DEPTHS };

  DelaunayDsoInitializer(DsoSystem *dsoSystem, CameraModel *cam,
                         DebugOutputType debugOutputType);

  // returns true if initialization is completed
  bool addFrame(const cv::Mat &frame, int globalFrameNum);

  std::vector<KeyFrame> createKeyFrames();

  static std::vector<KeyFrame> createKeyFramesDelaunay(
      CameraModel *cam, cv::Mat frames[2], int frameNums[2],
      StdVector<Vec2> initialPoints[2], std::vector<double> initialDepths[2],
      const SE3 &firstToSecond, DebugOutputType debugOutputType);

private:
  CameraModel *cam;
  DsoSystem *dsoSystem;
  StereoMatcher stereoMatcher;
  bool hasFirstFrame;
  int framesSkipped;
  cv::Mat frames[2];
  int globalFrameNums[2];
  DebugOutputType debugOutputType;
};

} // namespace fishdso

#endif
