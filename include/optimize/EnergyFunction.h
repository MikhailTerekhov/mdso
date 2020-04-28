#ifndef INCLUDE_ENERGYFUNCTION
#define INCLUDE_ENERGYFUNCTION

#include "Residual.h"
#include "optimize/DeltaParameterVector.h"
#include "optimize/FrameParameterOrder.h"
#include "optimize/Gradient.h"
#include "optimize/Hessian.h"
#include "optimize/Parameters.h"
#include "optimize/parametrizations.h"
#include "optimize/precomputations.h"
#include "util/types.h"
#include <optional>

namespace mdso::optimize {

class EnergyFunction {
public:
  class Values {
  public:
    Values(const StdVector<Residual> &residuals, const Parameters &parameters,
           const ceres::LossFunction *lossFunction,
           PrecomputedHostToTarget &hostToTarget,
           PrecomputedLightHostToTarget &lightHostToTarget);

    const VecRt &values(int residualInd) const;
    const Residual::CachedValues &cachedValues(int residualInd) const;
    double totalEnergy(const StdVector<Residual> &residuals) const;

  private:
    const ceres::LossFunction *lossFunction;
    StdVector<std::pair<VecRt, Residual::CachedValues>> valsAndCache;
  };

  struct Derivatives {
    Derivatives(const Parameters &parameters,
                const StdVector<Residual> &residuals, const Values &values,
                PrecomputedHostToTarget &hostToTarget,
                PrecomputedMotionDerivatives &motionDerivatives,
                PrecomputedLightHostToTarget &lightHostToTarget);

    StdVector<Residual::Jacobian> residualJacobians;
    Parameters::Jacobians parametrizationJacobians;
  };

  EnergyFunction(CameraBundle *camBundle, KeyFrame *keyFrames[],
                 int numKeyFrames, const EnergyFunctionSettings &settings);

  int numPoints() const;

  VecRt getResidualValues(int residualInd);
  VecRt getPredictedResidualIncrement(int residualInd,
                                      const DeltaParameterVector &delta);
  static T getPredictedDeltaEnergy(const Hessian &hessian,
                                   const Gradient &gradient,
                                   const DeltaParameterVector &delta);
  inline const StdVector<Residual> &getResiduals() const { return residuals; }
  inline T getLogDepth(const Residual &res) {
    return parameters->logDepth(res.pointInd());
  }

  PrecomputedHostToTarget precomputeHostToTarget() const;
  PrecomputedMotionDerivatives precomputeMotionDerivatives() const;
  PrecomputedLightHostToTarget precomputeLightHostToTarget() const;

  Values createValues(PrecomputedHostToTarget &hostToTarget,
                      PrecomputedLightHostToTarget &lightHostToTarget);
  Derivatives
  createDerivatives(const Values &values, PrecomputedHostToTarget &hostToTarget,
                    PrecomputedMotionDerivatives &motionDerivatives,
                    PrecomputedLightHostToTarget &lightHostToTarget);
  double totalEnergy();
  Hessian getHessian();
  Gradient getGradient();

  void precomputeValuesAndDerivatives();
  void clearPrecomputations();

  std::shared_ptr<Parameters> getParameters();

  Parameters::State saveState() const;
  void recoverState(const Parameters::State &oldState);

  void optimize(int maxInterations);

private:
  double predictEnergyViaJacobian(const DeltaParameterVector &delta);

  Values &computeValues(PrecomputedHostToTarget &hostToTarget,
                        PrecomputedLightHostToTarget &lightHostToTarget);
  Values &computeValues();
  Derivatives &
  computeDerivatives(PrecomputedHostToTarget &hostToTarget,
                     PrecomputedMotionDerivatives &motionDerivatives,
                     PrecomputedLightHostToTarget &lightHostToTarget);
  Derivatives &computeDerivatives();
  Hessian getHessian(const Values &precomputedValues,
                     const Derivatives &precomputedDerivatives);
  Gradient getGradient(const Values &precomputedValues,
                       const Derivatives &precomputedDerivatives);

  StdVector<Residual> residuals;
  std::optional<Values> values;
  std::optional<Derivatives> derivatives;
  std::shared_ptr<Parameters> parameters;
  std::unique_ptr<ceres::LossFunction> lossFunction;
  CameraBundle *cam;
  EnergyFunctionSettings settings;
};

} // namespace mdso::optimize

#endif
