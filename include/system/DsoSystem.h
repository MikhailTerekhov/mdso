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
  DsoSystem(const SnapshotLoader &snapshotLoader, const Observers &observers,
            const Settings &_settings);
  ~DsoSystem();

  std::shared_ptr<PreKeyFrame> addFrame(const cv::Mat &frame,
                                        int globalFrameNum);
  template <typename PointT>
  void projectOntoBaseKf(StdVector<Vec2> *points, std::vector<double> *depths,
                         std::vector<PointT *> *ptrs,
                         std::vector<KeyFrame *> *kfs);

  void addFrameTrackerObserver(FrameTrackerObserver *observer);

  void saveSnapshot(const std::string &snapshotDir) const;

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

  double getTimeLastByLbo();
  SE3 predictInternal(double timeLastByLbo, const SE3 &baseToLbo,
                      const SE3 &baseToLast);
  SE3 predictBaseKfToCur();
  SE3 purePredictBaseKfToCur();

  void adjustWorldToFrameSizes(int newFrameNum);

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

  StdMap<int, KeyFrame> keyFrames;

  std::vector<int> frameNumbers;
  StdVector<SE3> worldToFrame;
  StdVector<SE3> worldToFramePredict;

  AffineLightTransform<double> lightKfToLast;

  double lastTrackRmse;

  Settings settings;

  Observers observers;
};

} // namespace fishdso

#endif
