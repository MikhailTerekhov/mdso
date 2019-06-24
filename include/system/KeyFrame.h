#ifndef INCLUDE_KEYFRAME
#define INCLUDE_KEYFRAME

#include "system/ImmaturePoint.h"
#include "system/OptimizedPoint.h"
#include "system/PreKeyFrame.h"
#include "util/DepthedImagePyramid.h"
#include "util/PixelSelector.h"
#include "util/settings.h"
#include <Eigen/StdVector>
#include <memory>
#include <opencv2/core.hpp>
#include <vector>

namespace fishdso {

struct KeyFrame {
  KeyFrame(const KeyFrame &other) = delete;
  KeyFrame(KeyFrame &&other) = default;
  KeyFrame(CameraModel *cam, const cv::Mat &frameColored, int globalFrameNum,
           PixelSelector &pixelSelector,
           const Settings::KeyFrame &_kfSettings = {},
           const PointTracerSettings tracingSettings = {});

  KeyFrame(std::shared_ptr<PreKeyFrame> newPreKeyFrame,
           PixelSelector &pixelSelector,
           const Settings::KeyFrame &_kfSettings = {},
           const PointTracerSettings &tracingSettings = {});

  void activateAllImmature();
  void deactivateAllOptimized();

  void addImmatures(const std::vector<cv::Point> &points);

  void selectPointsDenser(PixelSelector &pixelSelector, int pointsNeeded);

  cv::Mat3b drawDepthedFrame(double minDepth, double maxDepth) const;

  std::shared_ptr<PreKeyFrame> preKeyFrame;

  std::vector<std::unique_ptr<ImmaturePoint>> immaturePoints;
  std::vector<std::unique_ptr<OptimizedPoint>> optimizedPoints;

  std::vector<std::shared_ptr<PreKeyFrame>> trackedFrames;

  Settings::KeyFrame kfSettings;
  PointTracerSettings tracingSettings;
};

} // namespace fishdso

#endif
