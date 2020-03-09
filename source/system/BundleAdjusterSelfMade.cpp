#include "system/BundleAdjusterSelfMade.h"

#include "optimize/EnergyFunction.h"

namespace mdso {

using namespace optimize;

BundleAdjusterSelfMade::~BundleAdjusterSelfMade() {}

void BundleAdjusterSelfMade::adjust(
    KeyFrame **keyFrames, int numKeyFrames,
    const BundleAdjusterSettings &settings) const {
  CHECK_GE(numKeyFrames, 2);
  CameraBundle *cam = keyFrames[0]->preKeyFrame->cam;
  EnergyFunction energyFunction(cam, keyFrames, numKeyFrames,
                                settings.energyFunction);
  energyFunction.optimize(settings.optimization.maxIterations);
}

} // namespace mdso