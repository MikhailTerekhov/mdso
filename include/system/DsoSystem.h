#ifndef INCLUDE_DSOSYSTEM
#define INCLUDE_DSOSYSTEM

#include "output/Observers.h"
#include "system/BundleAdjuster.h"
#include "system/CameraModel.h"
#include "system/DsoInitializer.h"
#include "system/FrameTracker.h"
#include "system/KeyFrame.h"
#include "system/MarginalizedKeyFrame.h"
#include "system/Preprocessor.h"
#include "util/DepthedImagePyramid.h"
#include "util/DistanceMap.h"
#include "util/PlyHolder.h"
#include "util/settings.h"
#include <map>
#include <memory>
#include <opencv2/core.hpp>
#include <variant>

namespace mdso {

class DsoSystem {
public:
  DsoSystem(CameraBundle *cam, Preprocessor *preprocessor,
            const Observers &observers = {}, const Settings &settings = {});
  ~DsoSystem();

  void addMultiFrame(const cv::Mat frames[], Timestamp timestamps[]);

  void addFrameTrackerObserver(FrameTrackerObserver *observer);

  template <typename PointT>
  void projectOntoFrame(int globalFrameNum, Vec2 *points[],
                        const std::optional<PointT ***> &refs,
                        const std::optional<int **> &pointIndices,
                        const std::optional<double **> &depths, int sizes[]);

private:
  inline KeyFrame &lastKeyFrame() { return *keyFrames.back(); }
  inline KeyFrame &lboKeyFrame() { return **(++keyFrames.rbegin()); }
  inline KeyFrame &baseFrame() {
    return settings.trackFromLastKf ? lastKeyFrame() : lboKeyFrame();
  }

  Timestamp getTimestamp(int frameNumber);
  SE3 getFrameToWorld(int frameNumber);
  AffLight getLightWorldToFrame(int frameNumber, int ind);

  double getTimeLastByLbo();
  SE3 predictInternal(double timeLastByLbo, const SE3 &baseToLBTwo,
                      const SE3 &baseToLBOne);
  SE3 predictBaseKfToCur();

  FrameTracker::TrackingResult predictTracking();

  int totalOptimized() const;

  bool doNeedKf(PreKeyFrame *lastFrame);
  void marginalizeFrames();
  
  void activateOptimizedRandom();
  void activateOptimizedDist();
  void activateNewOptimizedPoints();

  void traceOn(const PreKeyFrame &frame);

  CameraBundle *cam;
  CameraBundle::CamPyr camPyr;

  std::vector<PixelSelector> pixelSelector;

  std::unique_ptr<DsoInitializer> dsoInitializer;
  bool isInitialized;

  std::unique_ptr<FrameTracker> frameTracker;

  std::vector<std::unique_ptr<MarginalizedKeyFrame>> marginalizedFrames;
  std::vector<std::unique_ptr<KeyFrame>> keyFrames;

  std::vector<std::variant<MarginalizedKeyFrame *, MarginalizedPreKeyFrame *,
                           KeyFrame *, PreKeyFrame *>>
      allFrames;

  Settings settings;
  PointTracerSettings pointTracerSettings;

  Observers observers;

  Preprocessor *preprocessor;
};

} // namespace mdso

#endif
