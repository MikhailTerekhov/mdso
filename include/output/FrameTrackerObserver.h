#ifndef INCLUDE_FRAMETRACKEROBSERVER
#define INCLUDE_FRAMETRACKEROBSERVER

#include "system/FrameTracker.h"
#include <ceres/ceres.h>

namespace fishdso {

class FrameTrackerObserver {
public:
  virtual ~FrameTrackerObserver() = 0;

  virtual void newBaseFrame(const DepthedImagePyramid &pyr) {}
  virtual void startTracking(const ImagePyramid &frame) {}
  virtual void
  levelTracked(int pyrLevel, const SE3 &baseToLast,
               const AffineLightTransform<double> &affLightBaseToLast,
               const StdVector<std::pair<Vec2, double>> &pointResiduals) {}
};

} // namespace fishdso

#endif
