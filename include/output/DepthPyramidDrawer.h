#ifndef INCLUDE_DEPTHPYRAMIDDRAWER
#define INCLUDE_DEPTHPYRAMIDDRAWER

#include "output/FrameTrackerObserver.h"
#include "output/TrackingDebugImageDrawer.h"

DECLARE_double(pyr_rel_point_size);
DECLARE_int32(pyr_image_width);

namespace fishdso {

class DepthPyramidDrawer : public FrameTrackerObserver {
public:
  void newBaseFrame(const FrameTracker::DepthedMultiFrame &pyr) override;

  bool pyrChanged();
  cv::Mat getLastPyr();

private:
  cv::Mat lastPyr;
  bool mPyrChanged = false;
};

} // namespace fishdso

#endif
