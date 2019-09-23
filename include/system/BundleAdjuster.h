#ifndef INCLUDE_BUNDLEADJUSTER
#define INCLUDE_BUNDLEADJUSTER

#include "system/CameraModel.h"
#include "system/KeyFrame.h"
#include <sophus/se3.hpp>

namespace mdso {

class BundleAdjuster {
public:
  BundleAdjuster(CameraBundle *cam, KeyFrame *keyFrames[], int size,
                 const BundleAdjusterSettings &_settings);
  void adjust(int maxNumIterations);

private:
  CameraBundle *cam;
  KeyFrame **keyFrames;
  int size;

  BundleAdjusterSettings settings;
};

} // namespace mdso

#endif
