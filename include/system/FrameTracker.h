#ifndef INCLUDE_FRAMETRACKER
#define INCLUDE_FRAMETRACKER

#include "system/AffineLightTransform.h"
#include "system/CameraModel.h"
#include "system/KeyFrame.h"
#include "util/DepthedImagePyramid.h"

namespace fishdso {

class FrameTracker {
public:
  FrameTracker(const StdVector<CameraModel> &camPyr, const DepthedImagePyramid &baseFrame);

  std::pair<SE3, AffineLightTransform<double>>
  trackFrame(const ImagePyramid &frame, const SE3 &coarseMotion,
             const AffineLightTransform<double> &coarseAffLight);

private:
  std::pair<SE3, AffineLightTransform<double>>
  trackPyrLevel(const CameraModel &cam, const cv::Mat1b &baseImg,
                const cv::Mat1d &baseDepths, const cv::Mat1b &trackedImg,
                const SE3 &coarseMotion,
                const AffineLightTransform<double> &coarseAffLight);

  const StdVector<CameraModel> &camPyr;
  DepthedImagePyramid baseFrame;  
  int displayWidth, displayHeight;
};

} // namespace fishdso

#endif
