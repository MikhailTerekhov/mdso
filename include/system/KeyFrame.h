#ifndef INCLUDE_KEYFRAME
#define INCLUDE_KEYFRAME

#include "system/DsoInitializer.h"
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

struct KeyFrameEntry {
  static_vector<ImmaturePoint, Settings::KeyFrame::max_immaturePointsNum>
      immaturePoints;
  static_vector<OptimizedPoint, Settings::max_maxOptimizedPoints>
      optimizedPoints;

  AffLight lightWorldToThis;
};

struct KeyFrame {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  KeyFrame(const KeyFrame &other) = delete;
  KeyFrame(KeyFrame &&other) = delete;
  KeyFrame(const InitializedFrame &initializedFrame,
           CameraBundle *cam, int globalFrameNum, long long timestamp,
           PixelSelector pixelSelector[],
           const Settings::KeyFrame &_kfSettings = {},
           const Settings::Pyramid &pyrSettings = {},
           const PointTracerSettings &tracingSettings = {});
  KeyFrame(std::unique_ptr<PreKeyFrame> newPreKeyFrame,
           PixelSelector pixelSelector[],
           const Settings::KeyFrame &_kfSettings = {},
           const PointTracerSettings &tracingSettings = {});

  void selectPointsDenser(PixelSelector pixelSelector[], int pointsNeeded);

  std::unique_ptr<PreKeyFrame> preKeyFrame;
  SE3 thisToWorld;
  KeyFrameEntry frames[Settings::CameraBundle::max_camerasInBundle];
  static_vector<std::unique_ptr<PreKeyFrame>, Settings::max_keyFrameDist>
      trackedFrames;

  Settings::KeyFrame kfSettings;
  PointTracerSettings tracingSettings;

private:
  void addImmatures(const cv::Point points[], int size, int numInBundle);
};

} // namespace fishdso

#endif
