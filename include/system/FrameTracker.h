#ifndef INCLUDE_FRAMETRACKER
#define INCLUDE_FRAMETRACKER

#include "system/AffineLightTransform.h"
#include "system/CameraModel.h"
#include "system/KeyFrame.h"
#include "util/DepthedImagePyramid.h"

namespace fishdso {

class FrameTracker {
public:
  FrameTracker(const StdVector<CameraModel> &camPyr,
               std::unique_ptr<DepthedImagePyramid> baseFrame,
               const Settings::FrameTracker &frameTrackerSettings = {},
               const Settings::Pyramid &pyrSettings = {},
               const Settings::AffineLight &affineLightSettings = {},
               const Settings::Intencity &intencitySettings = {},
               const Settings::GradWeighting &gradWeightingSettings = {},
               const Settings::Threading &threadingSettings = {});

  std::pair<SE3, AffineLightTransform<double>>
  trackFrame(const ImagePyramid &frame, const SE3 &coarseMotion,
             const AffineLightTransform<double> &coarseAffLight);

  // output only
  std::vector<cv::Mat3b> residualsImg;

  double lastRmse;

private:
  std::pair<SE3, AffineLightTransform<double>>
  trackPyrLevel(const CameraModel &cam, const cv::Mat1b &baseImg,
                const cv::Mat1d &baseDepths, const cv::Mat1b &trackedImg,
                const SE3 &coarseMotion,
                const AffineLightTransform<double> &coarseAffLight,
                int pyrLevel);

  const StdVector<CameraModel> &camPyr;
  std::unique_ptr<DepthedImagePyramid> baseFrame;
  int displayWidth, displayHeight;

  Settings::FrameTracker frameTrackerSettings;
  Settings::Pyramid pyrSettings;
  Settings::AffineLight affineLightSettings;
  Settings::Intencity intencitySettings;
  Settings::GradWeighting gradWeightingSettings;
  Settings::Threading threadingSettings;
};

} // namespace fishdso

#endif
