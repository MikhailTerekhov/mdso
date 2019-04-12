#ifndef INCLUDE_DSOSYSTEM
#define INCLUDE_DSOSYSTEM

#include "output/Observers.h"
#include "system/BundleAdjuster.h"
#include "system/CameraModel.h"
#include "system/DsoInitializer.h"
#include "system/FrameTracker.h"
#include "system/KeyFrame.h"
#include "util/DepthedImagePyramid.h"
#include "util/DistanceMap.h"
#include "util/PlyHolder.h"
#include "util/settings.h"
#include <map>
#include <memory>
#include <opencv2/core.hpp>
#include <optional>

namespace fishdso {

class DsoSystem {
public:
  DsoSystem(CameraModel *cam, const Observers &observers = {},
            const Settings &settings = {});
  ~DsoSystem();

  std::shared_ptr<PreKeyFrame> addFrame(const cv::Mat &frame,
                                        int globalFrameNum);
  template <typename PointT>
  void projectOntoBaseKf(StdVector<Vec2> *points, std::vector<double> *depths,
                         std::vector<PointT *> *ptrs,
                         std::vector<KeyFrame *> *kfs);

  void addFrameTrackerObserver(FrameTrackerObserver *observer);

  // output only
  KeyFrame *lastInitialized;
  StdVector<std::pair<Vec2, double>> lastKeyPointDepths;

  double scaleGTToOur;
  SE3 gtToOur;

private:
  EIGEN_STRONG_INLINE KeyFrame &lastKeyFrame() {
    return keyFrames.rbegin()->second;
  }
  EIGEN_STRONG_INLINE KeyFrame &lboKeyFrame() {
    return (++keyFrames.rbegin())->second;
  }
  EIGEN_STRONG_INLINE KeyFrame &baseKeyFrame() {
    return settings.trackFromLastKf ? lastKeyFrame() : lboKeyFrame();
  }

  SE3 predictInternal(int prevFramesSkipped, const SE3 &worldToBaseKf,
                      const SE3 &worldToLbo, const SE3 &worldToLast);
  SE3 predictBaseKfToCur();
  SE3 purePredictBaseKfToCur();

  void checkLastTrackedStereo(PreKeyFrame *lastFrame);
  void checkLastTrackedGT(PreKeyFrame *lastFrame);

  bool didTrackFail();
  std::pair<SE3, AffineLightTransform<double>>
  recoverTrack(PreKeyFrame *lastFrame);

  bool doNeedKf(PreKeyFrame *lastFrame);
  void marginalizeFrames();
  void activateNewOptimizedPoints();

  CameraModel *cam;
  StdVector<CameraModel> camPyr;

  PixelSelector pixelSelector;

  std::unique_ptr<DsoInitializer> dsoInitializer;
  bool isInitialized;

  std::unique_ptr<FrameTracker> frameTracker;

  std::map<int, KeyFrame> keyFrames;
  StdMap<int, SE3> worldToFrame;
  StdMap<int, SE3> worldToFramePredict;
  StdMap<int, SE3> worldToFrameMatched;
  StdMap<int, SE3> worldToFrameGT;

  AffineLightTransform<double> lightKfToLast;

  double lastTrackRmse;

  int firstFrameNum;

  Settings settings;

  Observers observers;
};

} // namespace fishdso

#endif
