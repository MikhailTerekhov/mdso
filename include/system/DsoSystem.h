#ifndef INCLUDE_DSOSYSTEM
#define INCLUDE_DSOSYSTEM

#include "system/BundleAdjuster.h"
#include "system/CameraModel.h"
#include "system/DsoInitializer.h"
#include "system/FrameTracker.h"
#include "system/KeyFrame.h"
#include "util/DepthedImagePyramid.h"
#include "util/settings.h"
#include <map>
#include <memory>
#include <opencv2/core.hpp>
#include <bits/vector.tcc>

namespace fishdso {

class DsoSystem {
public:
  DsoSystem(CameraModel *cam);
  ~DsoSystem();

  std::shared_ptr<PreKeyFrame> addFrame(const cv::Mat &frame, int globalFrameNum);
  void addGroundTruthPose(int globalFrameNum, const SE3 &worldToThat);

  void printLastKfInPly(std::ostream &out);
  void printTrackingInfo(std::ostream &out);
  void printPredictionInfo(std::ostream &out);
  void printGroundTruthInfo(std::ostream &out);
  void printMatcherInfo(std::ostream &out);

  // output only
  KeyFrame *lastInitialized;
  StdVector<std::pair<Vec2, double>> lastKeyPointDepths;

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

  std::unique_ptr<DepthedImagePyramid> optimizedPointsOntoBaseKf();

  SE3 predictBaseKfToCur();
  SE3 purePredictBaseKfToCur();

  void checkLastTrackedStereo(PreKeyFrame *lastFrame);
  void checkLastTrackedGT(PreKeyFrame *lastFrame);

  bool checkNeedKf(PreKeyFrame *lastFrame);
  void marginalizeFrames();
  void activateNewOptimizedPoints();

  static void printMotionInfo(std::ostream &out,
                              const StdMap<int, SE3> &motions);

  CameraModel *cam;
  StdVector<CameraModel> camPyr;

  std::unique_ptr<DsoInitializer> dsoInitializer;
  bool isInitialized;

  std::unique_ptr<FrameTracker> frameTracker;

  StdVector<std::shared_ptr<PreKeyFrame>> frameHistory;
  std::map<int, KeyFrame> keyFrames;
  StdMap<int, SE3> worldToFrame;
  StdMap<int, SE3> worldToFramePredict;
  StdMap<int, SE3> worldToFrameMatched;
  StdMap<int, SE3> worldToFrameGT;

  AffineLightTransform<double> lightKfToLast;
};

} // namespace fishdso

#endif
