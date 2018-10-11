#ifndef INCLUDE_FRAMETRACKER
#define INCLUDE_FRAMETRACKER

#include "system/AffineLightTransform.h"
#include "system/CameraModel.h"
#include "system/KeyFrame.h"

namespace fishdso {

class FrameTracker {
public:
  FrameTracker(const StdVector<CameraModel> &camPyr, PreKeyFrame *base);

  std::pair<SE3, AffineLightTransform<double>>
  trackFrame(PreKeyFrame *frame, const SE3 &coarseMotion,
             const AffineLightTransform<double> &coarseAffLight);

private:
  std::pair<SE3, AffineLightTransform<double>>
  trackPyrLevel(const CameraModel &cam, const cv::Mat1b &baseImg,
                const cv::Mat1d &baseDepths, const cv::Mat1b &trackedImg,
                const SE3 &coarseMotion,
                const AffineLightTransform<double> &coarseAffLight);

  const StdVector<CameraModel> &camPyr;
  PreKeyFrame *base;
};

} // namespace fishdso

#endif
