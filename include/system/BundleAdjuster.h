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
  BundleAdjuster(CameraModel *cam);

  void addKeyFrame(KeyFrame *keyFrame);
  void adjust();

private:
  bool isOOB(const SE3 &worldToBase, const SE3 &worldToRef,
             const OptimizedPoint &baseOP);
  CameraModel *cam;
  std::set<KeyFrame *> keyFrames;
  KeyFrame *firstKeyFrame;
};

} // namespace fishdso

#endif
