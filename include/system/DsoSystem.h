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
  void printMatcherInfo(std::ostream &out);

private:
  EIGEN_STRONG_INLINE KeyFrame &lastKeyFrame() {
    return keyFrames.rbegin()->second;
  }
  EIGEN_STRONG_INLINE KeyFrame &lboKeyFrame() {
    return (++keyFrames.rbegin())->second;
  }
  EIGEN_STRONG_INLINE KeyFrame &baseKeyFrame() {
    return FLAGS_track_from_lask_kf ? lastKeyFrame() : lboKeyFrame();
  }

  SE3 predictBaseKfToCur();
  SE3 purePredictBaseKfToCur();

  void checkLastTracked(std::unique_ptr<PreKeyFrame> lastFrame);

  static void printMotionInfo(std::ostream &out,
                              const StdMap<int, SE3> &motions);

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
  StdMap<int, SE3> worldToFrameMatched;

  AffineLightTransform<double> lightKfToLast;
};

} // namespace fishdso

#endif
