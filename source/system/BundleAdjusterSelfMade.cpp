#include "system/BundleAdjusterSelfMade.h"
#include "internal/optimize/EnergyFunctionCeres.h"
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
  switch (settings.optimization.optimizationType) {
  case Settings::Optimization::DISABLED:
  case Settings::Optimization::CERES:
    LOG(ERROR) << "Unexpected optimization type";
  case Settings::Optimization::SELF_WRITTEN:
    energyFunction.optimize(settings.optimization.maxIterations);
    break;
  case Settings::Optimization::MIXED:
    EnergyFunctionCeres energyFunctionCeres(energyFunction, settings);
    energyFunctionCeres.optimize();
    energyFunctionCeres.applyParameterUpdate();
  }
}

} // namespace mdso