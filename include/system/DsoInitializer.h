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
  enum DebugOutputType { NO_DEBUG, SPARSE_DEPTHS, FILLED_DEPTHS };

  DsoInitializer(CameraModel *cam);

  // returns true if initialization is completed
  bool addFrame(const cv::Mat &frame, int globalFrameNum);

  std::vector<KeyFrame> createKeyFrames(DebugOutputType debugOutputType);

  static std::vector<KeyFrame> createKeyFramesDelaunay(
      CameraModel *cam, cv::Mat frames[2], int frameNums[2],
      StdVector<Vec2> initialPoints[2], std::vector<double> initialDepths[2],
      const SE3 &firstToSecond, DebugOutputType debugOutputType);

private:
  std::vector<KeyFrame>
  createKeyFramesFromStereo(DebugOutputType debugOutputType);
  std::vector<KeyFrame> createKeyFramesDummy();

  CameraModel *cam;
  StereoMatcher stereoMatcher;
  bool hasFirstFrame;
  int framesSkipped;
  cv::Mat frames[2];
  int globalFrameNums[2];
};

} // namespace fishdso

#endif
