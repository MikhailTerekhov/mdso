#ifndef INCLUDE_STEPCONTROLLER
#define INCLUDE_STEPCONTROLLER

#include "util/settings.h"

namespace mdso::optimize {

class StepController {
public:
  StepController(const Settings::Optimization::StepControl &newSettings);

  bool newStep(double oldEnergy, double newEnergy, double predictedEnergy);
  double lambda() const;

private:
  double mLambda;
  double failMultiplier;
  Settings::Optimization::StepControl settings;
};

} // namespace mdso::optimize

#endif
