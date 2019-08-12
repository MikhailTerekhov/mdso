#ifndef INCLUDE_DSOSYSTEM
#define INCLUDE_DSOSYSTEM

#include "output/Observers.h"
#include "system/BundleAdjuster.h"
#include "system/CameraModel.h"
#include "system/DsoInitializer.h"
#include "system/FrameTracker.h"
#include "system/KeyFrame.h"
#include "system/MarginalizedKeyFrame.h"
#include "util/DepthedImagePyramid.h"
#include "util/DistanceMap.h"
#include "util/PlyHolder.h"
#include "util/settings.h"
#include <map>
#include <memory>
#include <opencv2/core.hpp>
#include <variant>

namespace fishdso {

class DsoSystem {
public:
  DsoSystem(CameraBundle *cam, const Observers &observers = {},
            const Settings &settings = {});
  ~DsoSystem();

  void addMultiFrame(const cv::Mat frames[], long long timestamps[]);

  void addFrameTrackerObserver(FrameTrackerObserver *observer);

  template <typename PointT>
  void projectOntoBaseKf(Vec2 *points[], const std::optional<PointT ***> &refs,
                         const std::optional<int **> &pointIndices,
                         const std::optional<double **> &depths, int sizes[]);

private:
  inline KeyFrame &lastKeyFrame() { return *keyFrames.back(); }
  inline KeyFrame &lboKeyFrame() { return **(++keyFrames.rbegin()); }
  inline KeyFrame &baseFrame() {
    return settings.trackFromLastKf ? lastKeyFrame() : lboKeyFrame();
  }

  long long getTimestamp(int frameNumber);
  SE3 getFrameToWorld(int frameNumber);
  AffLight getLightWorldToFrame(int frameNumber, int ind);

  double getTimeLastByLbo();
  SE3 predictInternal(double timeLastByLbo, const SE3 &baseToLbo,
                      const SE3 &baseToLast);
  SE3 predictBaseKfToCur();
  SE3 purePredictBaseKfToCur();

  FrameTracker::TrackingResult predictTracking();

  bool doNeedKf(PreKeyFrame *lastFrame);
  void marginalizeFrames();
  void activateNewOptimizedPoints();

  void traceOn(const PreKeyFrame &frame);

  CameraBundle *cam;
  CameraBundle::CamPyr camPyr;

  std::vector<PixelSelector> pixelSelector;

  std::unique_ptr<DsoInitializer> dsoInitializer;
  bool isInitialized;

  std::unique_ptr<FrameTracker> frameTracker;

  std::vector<std::unique_ptr<MarginalizedKeyFrame>> marginalizedFrames;
  static_vector<std::unique_ptr<KeyFrame>, Settings::max_maxKeyFrames>
      keyFrames;

  std::vector<std::variant<MarginalizedKeyFrame *, MarginalizedPreKeyFrame *,
                           KeyFrame *, PreKeyFrame *>>
      allFrames;

  Settings settings;

  Observers observers;
};

} // namespace fishdso

#endif
