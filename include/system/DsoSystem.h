#ifndef INCLUDE_DSOSYSTEM
#define INCLUDE_DSOSYSTEM

#include "system/BundleAdjuster.h"
#include "system/CameraModel.h"
#include "system/DsoInitializer.h"
#include "system/FrameTracker.h"
#include "system/KeyFrame.h"
#include "util/settings.h"
#include <map>
#include <memory>
#include <opencv2/core.hpp>

namespace fishdso {

class DsoSystem {
public:
  DsoSystem(CameraModel *cam);
  ~DsoSystem();

  void addFrame(const cv::Mat &frame);

  void printLastKfInPly(std::ostream &out);
  void printTrackingInfo(std::ostream &out);
  void printPredictionInfo(std::ostream &out);

private:
  SE3 predictKfToCur();
  SE3 purePredictKfToCur();

  CameraModel *cam;
  StdVector<CameraModel> camPyr;

  DsoInitializer dsoInitializer;
  bool isInitialized;

  std::unique_ptr<FrameTracker> frameTracker;
  std::unique_ptr<BundleAdjuster> bundleAdjuster;

  int curFrameNum;
  std::map<int, KeyFrame> keyFrames;
  StdMap<int, SE3> worldToFrame;
  StdMap<int, SE3> worldToFramePredict;

  AffineLightTransform<double> lightKfToLast;
};

} // namespace fishdso

#endif
