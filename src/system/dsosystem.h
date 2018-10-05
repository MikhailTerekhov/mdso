#pragma once

#include "util/settings.h"
#include "system/cameramodel.h"
#include "system/dsoinitializer.h"
#include "system/frametracker.h"
#include "system/keyframe.h"
#include <map>
#include <memory>
#include <opencv2/core.hpp>

namespace fishdso {

class DsoSystem {
public:
  DsoSystem(CameraModel *cam);

  void addFrame(const cv::Mat &frame);

  void printLastKfInPly(std::ostream &out);
  void printTrackingInfo(std::ostream &out);
  void printPredictionInfo(std::ostream &out);

private:
  SE3 predictKfToCur();
  SE3 purePredictKfToCur();

  CameraModel *cam;
  stdvectorCameraModel camPyr;

  DsoInitializer dsoInitializer;
  bool isInitialized;
  
  std::unique_ptr<FrameTracker> frameTracker;
  
  int curFrameNum;
  std::map<int, KeyFrame> keyFrames;
  stdmapIntSE3 worldToFrame;
  stdmapIntSE3 worldToFramePredict;

  AffineLightTransform<double> lightKfToLast;
};

} // namespace fishdso
