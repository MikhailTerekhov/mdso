#ifndef INCLUDE_TRACKINGDEBUGIMAGEOBSERVER
#define INCLUDE_TRACKINGDEBUGIMAGEOBSERVER

#include "output/FrameTrackerObserver.h"

DECLARE_double(tracking_rel_point_size);
DECLARE_int32(tracking_res_image_width);
DECLARE_double(debug_max_residual);

namespace fishdso {

class TrackingDebugImageDrawer : public FrameTrackerObserver {
public:
  TrackingDebugImageDrawer(const StdVector<CameraModel> &camPyr,
                           const Settings::FrameTracker &frameTrackerSettings,
                           const Settings::Pyramid &pyrSettings);

  void startTracking(const ImagePyramid &frame);
  void levelTracked(int pyrLevel, const SE3 &baseToLast,
                    const AffLight &affLightBaseToLast,
                    const StdVector<std::pair<Vec2, double>> &pointResiduals);

  cv::Mat3b drawAllLevels();
  cv::Mat3b drawFinestLevel();

private:
  Settings::FrameTracker frameTrackerSettings;
  Settings::Pyramid pyrSettings;

  StdVector<CameraModel> camPyr;
  std::vector<cv::Mat1b> curFramePyr;
  std::vector<cv::Mat3b> residualsImg;
};

} // namespace fishdso

#endif
