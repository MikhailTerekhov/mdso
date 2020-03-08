#ifndef INCLUDE_BUNDLEADJUSTERCERES
#define INCLUDE_BUNDLEADJUSTERCERES

#include "system/CameraModel.h"
#include "system/KeyFrame.h"
#include <sophus/se3.hpp>

namespace mdso {

class BundleAdjusterCeres {
public:
  BundleAdjusterCeres(CameraBundle *cam, KeyFrame *keyFrames[], int size,
                      const BundleAdjusterSettings &_settings);
  void adjust(int maxNumIterations);

private:
  CameraBundle *cam;
  KeyFrame **keyFrames;
  StdVector<SE3> bodyToWorld;
  int size;

  BundleAdjusterSettings settings;
};

} // namespace mdso

#endif
