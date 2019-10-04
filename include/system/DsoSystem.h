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
#include "system/TrackingPredictor.h"
#include "util/DepthedImagePyramid.h"
#include "util/DistanceMap.h"
#include "util/PlyHolder.h"
#include "util/TrajectoryHolder.h"
#include "util/settings.h"
#include <map>
#include <memory>
#include <opencv2/core.hpp>
#include <variant>

namespace mdso {

class DsoSystem : public TrajectoryHolder {
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

  int trajectorySize() const override;
  int camNumber() const override;
  Timestamp timestamp(int ind) const override;
  SE3 bodyToWorld(int ind) const override;
  AffLight affLightWorldToBody(int ind, int camInd) const override;

private:
  inline KeyFrame &lastKeyFrame() { return *keyFrames.back(); }
  inline KeyFrame &lboKeyFrame() { return **(++keyFrames.rbegin()); }
  inline KeyFrame &baseFrame() {
    return settings.trackFromLastKf ? lastKeyFrame() : lboKeyFrame();
  }

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

  std::unique_ptr<TrackingPredictor> trackingPredictor;
};

} // namespace mdso

#endif
