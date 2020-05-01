#include "optimize/StepController.h"

namespace mdso::optimize {

StepController::StepController(
    const Settings::Optimization::StepControl &newSettings)
    : mLambda(newSettings.initialLambda)
    , failMultiplier(2.0)
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
  double predictionQuality = actualDiff / predictedDiff;
  double q2m1 = 2 * predictionQuality - 1;
  if (predictionQuality > 0) {
    mLambda *= std::max(1.0 / 3.0, q2m1 * q2m1 * q2m1);
    failMultiplier = 2;
  } else {
    mLambda *= failMultiplier;
    failMultiplier *= 2;
  }
  LOG(INFO) << "lambda: " << oldLambda << " -> " << mLambda;

  return predictionQuality > settings.acceptedRelDifference;
}

double StepController::lambda() const { return mLambda; }

} // namespace mdso::optimize