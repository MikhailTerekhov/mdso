#ifndef INCLUDE_FRAMETRACKER
#define INCLUDE_FRAMETRACKER

#include "system/AffineLightTransform.h"
#include "system/CameraModel.h"
#include "system/KeyFrame.h"
#include "util/DepthedImagePyramid.h"

namespace fishdso {

class FrameTrackerObserver;

class FrameTracker {
public:
  using DepthedMultiFrame = std::vector<DepthedImagePyramid>;

  struct TrackingResult {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    TrackingResult(int camNumber);

    SE3 baseToTracked;
    std::vector<AffLight> lightBaseToTracked;
  };

  FrameTracker(CameraBundle camPyr[], const DepthedMultiFrame &_baseFrame,
               std::vector<FrameTrackerObserver *> &observers,
               const FrameTrackerSettings &_settings = {});

  TrackingResult trackFrame(const PreKeyFrame &frame,
                            const TrackingResult &coarseTrackingResult);

private:
  TrackingResult trackPyrLevel(const PreKeyFrame &frame,
                               const TrackingResult &coarseTrackingResult,
                               int pyrLevel);

  CameraBundle *camPyr;
  DepthedMultiFrame baseFrame;
  std::vector<FrameTrackerObserver *> &observers;
  FrameTrackerSettings settings;
};

} // namespace fishdso

#endif
