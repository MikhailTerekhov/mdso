#ifndef INCLUDE_BUNDLEADJUSTER
#define INCLUDE_BUNDLEADJUSTER

#include "system/KeyFrame.h"

namespace mdso {

class BundleAdjuster {
public:
  virtual ~BundleAdjuster() = 0;

  virtual void adjust(KeyFrame **keyFrames, int numKeyFrames,
                      const BundleAdjusterSettings &settings) const = 0;
};

} // namespace mdso

#endif