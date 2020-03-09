#ifndef INCLUDE_BUNDLEADJUSTERSELFMADE
#define INCLUDE_BUNDLEADJUSTERSELFMADE

#include "system/BundleAdjuster.h"

namespace mdso {

class BundleAdjusterSelfMade : public BundleAdjuster {
public:
  ~BundleAdjusterSelfMade();

  void adjust(KeyFrame **keyFrames, int numKeyFrames,
              const BundleAdjusterSettings &settings) const override;
};

} // namespace mdso

#endif
