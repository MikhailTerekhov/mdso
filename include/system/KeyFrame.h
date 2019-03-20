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
           const Settings::KeyFrame &kfSettings = {},
           const Settings::PointTracer &tracingSettings = {},
           const Settings::Intencity &intencitySettings = {},
           const Settings::ResidualPattern &rpSettings = {},
           const Settings::Pyramid &pyrSettings = {});

  KeyFrame(std::shared_ptr<PreKeyFrame> newPreKeyFrame,
           PixelSelector &pixelSelector,
           const Settings::KeyFrame &kfSettings = {},
           const Settings::PointTracer &tracingSettings = {},
           const Settings::Intencity &intencitySettings = {},
           const Settings::ResidualPattern &rpSettings = {},
           const Settings::Pyramid &pyrSettings = {});

  void activateAllImmature();
  void deactivateAllOptimized();

  void addImmatures(const std::vector<cv::Point> &points);

  std::unique_ptr<DepthedImagePyramid> makePyramid();
  void selectPointsDenser(PixelSelector &pixelSelector, int pointsNeeded);

  cv::Mat3b drawDepthedFrame(double minDepth, double maxDepth) const;

  std::shared_ptr<PreKeyFrame> preKeyFrame;

  StdUnorderedSetOfPtrs<ImmaturePoint> immaturePoints;
  StdUnorderedSetOfPtrs<OptimizedPoint> optimizedPoints;

  Settings::KeyFrame kfSettings;
  Settings::PointTracer tracingSettings;
  Settings::Intencity intencitySettings;
  Settings::ResidualPattern rpSettings;
  Settings::Pyramid pyrSettings;
};

} // namespace fishdso

#endif
