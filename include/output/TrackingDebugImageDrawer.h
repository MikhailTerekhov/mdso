#ifndef INCLUDE_TRACKINGDEBUGIMAGEOBSERVER
#define INCLUDE_TRACKINGDEBUGIMAGEOBSERVER

#include "output/FrameTrackerObserver.h"

DECLARE_double(tracking_rel_point_size);
DECLARE_int32(tracking_res_image_width);
DECLARE_double(debug_max_residual);

namespace mdso {

class TrackingDebugImageDrawer : public FrameTrackerObserver {
public:
  TrackingDebugImageDrawer(CameraBundle camPyr[],
                           const Settings::FrameTracker &frameTrackerSettings,
                           const Settings::Pyramid &pyrSettings);

  void startTracking(const PreKeyFrame &frame) override;
  void levelTracked(int pyrLevel, const FrameTracker::TrackingResult &result,
                    std::pair<Vec2, double> pointResiduals[],
                    int size) override;

  cv::Mat3b drawAllLevels();
  cv::Mat3b drawFinestLevel();

private:
  Settings::FrameTracker frameTrackerSettings;
  Settings::Pyramid pyrSettings;

  CameraBundle *camPyr;
  std::vector<cv::Mat1b> curFramePyr;
  std::vector<cv::Mat3b> residualsImg;
};

} // namespace mdso

#endif
