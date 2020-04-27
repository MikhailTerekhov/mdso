#include "system/BundleAdjusterCeres.h"
#include "internal/optimize/EnergyFunctionCeres.h"

namespace mdso {

BundleAdjusterCeres::~BundleAdjusterCeres() {}

void BundleAdjusterCeres::adjust(KeyFrame *keyFrames[], int numKeyFrames,
                                 const BundleAdjusterSettings &settings) const {
  optimize::EnergyFunctionCeres energyFunction(keyFrames, numKeyFrames,
                                               settings);
  energyFunction.optimize();
  energyFunction.applyParameterUpdate();
}

} // namespace mdso
