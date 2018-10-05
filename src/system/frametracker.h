#pragma once

#include "system/affinelighttransform.h"
#include "system/cameramodel.h"
#include "system/keyframe.h"

namespace fishdso {

extern int dbg1;

class FrameTracker {
public:
  FrameTracker(const stdvectorCameraModel &camPyr, PreKeyFrame *base);

  std::pair<SE3, AffineLightTransform<double>>
  trackFrame(PreKeyFrame *frame, const SE3 &coarseMotion,
             const AffineLightTransform<double> &coarseAffLight);

private:
  std::pair<SE3, AffineLightTransform<double>>
  trackPyrLevel(const CameraModel &cam, const cv::Mat1b &baseImg,
                const cv::Mat1d &baseDepths, const cv::Mat1b &trackedImg,
                const SE3 &coarseMotion,
                const AffineLightTransform<double> &coarseAffLight);

  const stdvectorCameraModel &camPyr;
  PreKeyFrame *base;
};

} // namespace fishdso
