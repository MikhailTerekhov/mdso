#ifndef INCLUDE_FRAMETRACKER
#define INCLUDE_FRAMETRACKER

#include "system/AffineLightTransform.h"
#include "system/CameraModel.h"
#include "system/KeyFrame.h"
#include "util/DepthedImagePyramid.h"

namespace fishdso {

class FrameTrackerObserver;

class FrameTracker {
public:
  FrameTracker(const StdVector<CameraModel> &camPyr,
               std::unique_ptr<DepthedImagePyramid> _baseFrame,
               const std::vector<FrameTrackerObserver *> &observers = {},
               const FrameTrackerSettings &_settings = {});

  std::pair<SE3, AffineLightTransform<double>>
  trackFrame(const PreKeyFrame &frame, const SE3 &coarseBaseToTracked,
             const AffineLightTransform<double> &coarseAffLight);

  void addObserver(FrameTrackerObserver *observer);

  // output only
  std::vector<cv::Mat3b> residualsImg;

  double lastRmse;

private:
  std::pair<SE3, AffineLightTransform<double>>
  trackPyrLevel(const CameraModel &cam, const cv::Mat1b &baseImg,
                const cv::Mat1d &baseDepths, const cv::Mat1b &trackedImg,
                const PreKeyFrameInternals &trackedImgInternals,
                const SE3 &coarseBaseToTracked,
                const AffineLightTransform<double> &coarseAffLight,
                int pyrLevel);

  const StdVector<CameraModel> &camPyr;
  std::unique_ptr<DepthedImagePyramid> baseFrame;
  int displayWidth, displayHeight;

  std::vector<FrameTrackerObserver *> observers;
  FrameTrackerSettings settings;
};

} // namespace fishdso

#endif
