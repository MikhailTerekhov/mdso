#include "optimize/StepController.h"

namespace mdso::optimize {

StepController::StepController(
    const Settings::Optimization::StepControl &newSettings)
    : mLambda(newSettings.initialLambda)
    , settings(newSettings) {}

bool StepController::newStep(double oldEnergy, double newEnergy,
                             double predictedEnergy) {
  double predictedDiff = oldEnergy - predictedEnergy;
  double actualDiff = oldEnergy - newEnergy;
  LOG(INFO) << "actualDiff = " << actualDiff
            << " predictedDiff = " << predictedDiff
            << " relative diff = " << actualDiff / predictedDiff;
  if (predictedDiff < 0) {
    predictedDiff *= -1;
    actualDiff *= -1;
  }

  double oldLambda = mLambda;
  if (actualDiff > settings.goodRelDifference * predictedDiff)
    mLambda *= settings.successMultiplier;
  if (actualDiff < settings.badRelDifference * predictedDiff)
    mLambda *= settings.failMultiplier;
  LOG(INFO) << "lambda: " << oldLambda << " -> " << mLambda;

  return actualDiff > settings.acceptedRelDifference * predictedDiff;
}

double StepController::lambda() const { return mLambda; }

} // namespace mdso::optimize