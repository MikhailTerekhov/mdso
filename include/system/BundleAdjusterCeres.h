#ifndef INCLUDE_BUNDLEADJUSTERCERES
#define INCLUDE_BUNDLEADJUSTERCERES

#include "system/BundleAdjuster.h"

namespace mdso {

class BundleAdjusterCeres : public BundleAdjuster {
public:
  ~BundleAdjusterCeres();

  void adjust(KeyFrame *keyFrames[], int numKeyFrames,
              const BundleAdjusterSettings &settings) const override;
};

} // namespace mdso

#endif
