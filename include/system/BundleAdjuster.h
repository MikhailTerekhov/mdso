#ifndef INCLUDE_BUNDLEADJUSTER
#define INCLUDE_BUNDLEADJUSTER

#include "system/CameraModel.h"
#include "system/KeyFrame.h"
#include <memory>
#include <set>
#include <sophus/se3.hpp>

namespace fishdso {

class BundleAdjuster {
public:
  BundleAdjuster(CameraModel *cam,
                 const Settings::BundleAdjuster &bundleAdjusterSettings = {},
                 const Settings::ResidualPattern &rpSettings = {},
                 const Settings::GradWeighting &gradWeightingSettings = {},
                 const Settings::Intencity &intencitySettings = {},
                 const Settings::AffineLight &affineLightSettings = {},
                 const Settings::Threading &threadingSettings = {},
                 const Settings::Depth &depthSettings = {});

  void addKeyFrame(KeyFrame *keyFrame);
  void adjust(int maxNumIterations);

private:
  bool isOOB(const SE3 &worldToBase, const SE3 &worldToRef,
             const OptimizedPoint &baseOP);
  CameraModel *cam;
  std::set<KeyFrame *> keyFrames;
  KeyFrame *firstKeyFrame;

  Settings::BundleAdjuster bundleAdjusterSettings;
  Settings::ResidualPattern rpSettings;
  Settings::GradWeighting gradWeightingSettings;
  Settings::Intencity intencitySettings;
  Settings::AffineLight affineLightSettings;
  Settings::Threading threadingSettings;
  Settings::Depth depthSettings;
};

} // namespace fishdso

#endif
