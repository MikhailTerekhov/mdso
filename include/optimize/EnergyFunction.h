#ifndef INCLUDE_ENERGYFUNCTION
#define INCLUDE_ENERGYFUNCTION

#include "Residual.h"
#include "optimize/DeltaParameterVector.h"
#include "optimize/FrameParameterOrder.h"
#include "optimize/Gradient.h"
#include "optimize/Hessian.h"
#include "optimize/Parameters.h"
#include "optimize/parametrizations.h"
#include "util/types.h"
#include <optional>

namespace mdso::optimize {

class EnergyFunction {
public:
  EnergyFunction(CameraBundle *camBundle, KeyFrame *keyFrames[],
                 int numKeyFrames, const EnergyFunctionSettings &settings);

  int numPoints() const;

  VecRt getResidualValues(int residualInd);
  VecRt getPredictedResidualIncrement(int residualInd,
                                      const DeltaParameterVector &delta);
  inline const StdVector<Residual> &getResiduals() const { return residuals; }
  inline T getLogDepth(const Residual &res) {
    return parameters.logDepth(res.pointInd());
  }

  double totalEnergy();
  Hessian getHessian();
  Gradient getGradient();

  void precomputeValuesAndDerivatives();
  void clearPrecomputations();

  Parameters::State saveState() const;
  void recoverState(const Parameters::State &oldState);

  void optimize(int maxInterations);

private:
  class PrecomputedHostToTarget {
  public:
    PrecomputedHostToTarget(CameraBundle *cam, const Parameters *parameters);

    SE3t get(int hostInd, int hostCamInd, int targetInd, int targetCamInd);

  private:
    const Parameters *parameters;
    StdVector<SE3t> camToBody;
    StdVector<SE3t> bodyToCam;
    Array4d<SE3t> hostToTarget;
  };

  class PrecomputedMotionDerivatives {
  public:
    PrecomputedMotionDerivatives(CameraBundle *cam,
                                 const Parameters *parameters);
    const MotionDerivatives &get(int hostInd, int hostCamInd, int targetInd,
                                 int targetCamInd);

  private:
    const Parameters *parameters;
    StdVector<SE3t> camToBody;
    StdVector<SE3t> bodyToCam;
    Array4d<std::optional<MotionDerivatives>> hostToTargetDiff;
  };

  class PrecomputedLightHostToTarget {
  public:
    PrecomputedLightHostToTarget(const Parameters *parameters);

    AffLightT get(int hostInd, int hostCamInd, int targetInd, int targetCamInd);

  private:
    const Parameters *parameters;
    Array4d<std::optional<AffLightT>> lightHostToTarget;
  };

  class Values {
  public:
    Values(const StdVector<Residual> &residuals, const Parameters &parameters,
           const ceres::LossFunction *lossFunction,
           PrecomputedHostToTarget &hostToTarget,
           PrecomputedLightHostToTarget &lightHostToTarget);

    const VecRt &values(int residualInd) const;
    const Residual::CachedValues &cachedValues(int residualInd) const;
    double totalEnergy() const;

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

  PrecomputedHostToTarget precomputeHostToTarget() const;
  PrecomputedMotionDerivatives precomputeMotionDerivatives() const;
  PrecomputedLightHostToTarget precomputeLightHostToTarget() const;

  double predictEnergy(const DeltaParameterVector &delta);

  Values createValues(PrecomputedHostToTarget &hostToTarget,
                      PrecomputedLightHostToTarget &lightHostToTarget);
  Values &computeValues(PrecomputedHostToTarget &hostToTarget,
                        PrecomputedLightHostToTarget &lightHostToTarget);
  Values &computeValues();
  Derivatives
  createDerivatives(const Values &values, PrecomputedHostToTarget &hostToTarget,
                    PrecomputedMotionDerivatives &motionDerivatives,
                    PrecomputedLightHostToTarget &lightHostToTarget);
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
  Parameters parameters;
  std::unique_ptr<ceres::LossFunction> lossFunction;
  CameraBundle *cam;
  EnergyFunctionSettings settings;
};

} // namespace mdso::optimize

#endif
