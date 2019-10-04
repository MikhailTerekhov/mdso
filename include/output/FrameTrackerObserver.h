#ifndef INCLUDE_FRAMETRACKEROBSERVER
#define INCLUDE_FRAMETRACKEROBSERVER

#include "system/FrameTracker.h"
#include <ceres/ceres.h>

namespace mdso {

class FrameTrackerObserver {
public:
  virtual ~FrameTrackerObserver() = 0;

  virtual void newBaseFrame(const FrameTracker::DepthedMultiFrame &frame) {}
  virtual void startTracking(const PreKeyFrame &frame) {}
  virtual void levelTracked(
      int pyrLevel, const TrackingResult &result,
      const std::vector<StdVector<std::pair<Vec2, double>>> &pointResiduals) {}
};

} // namespace mdso

#endif
