#ifndef INCLUDE_DEPTHPYRAMIDDRAWER
#define INCLUDE_DEPTHPYRAMIDDRAWER

#include "output/FrameTrackerObserver.h"
#include "output/TrackingDebugImageDrawer.h"

DECLARE_double(pyr_rel_point_size);
DECLARE_int32(pyr_image_width);

namespace mdso {

class DepthPyramidDrawer : public FrameTrackerObserver {
public:
  void newBaseFrame(const FrameTracker::DepthedMultiFrame &pyr) override;

  bool pyrChanged();
  cv::Mat3b getLastPyr();

private:
  cv::Mat3b lastPyr;
  bool mPyrChanged = false;
};

} // namespace mdso

#endif
