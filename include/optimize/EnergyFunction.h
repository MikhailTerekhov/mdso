#ifndef INCLUDE_ENERGYFUNCTION
#define INCLUDE_ENERGYFUNCTION

#include "Residual.h"
#include "optimize/parametrizations.h"
#include "util/types.h"
#include <optional>

namespace mdso::optimize {

class EnergyFunction {
public:
  struct Gradient {
    VecXt frame;
    VecXt point;
  };

  struct Hessian {
    Hessian(int frameParams, int pointParams,
            const Settings::Optimization &settings);

    Hessian levenbergMarquardtDamp(double lambda) const;
    void solve(const Gradient &gradient, VecXt &deltaFrame, VecXt &deltaPoint,
               T lambda) const;

    MatXXt frameFrame;
    MatXXt framePoint;
    VecXt pointPoint;
    Settings::Optimization settings;
  };

  EnergyFunction(CameraBundle *camBundle, KeyFrame *keyFrames[],
                 int numKeyFrames, const EnergyFunctionSettings &settings);

  int numPoints() const;

  VecRt getResidualValues(int residualInd);
  inline const StdVector<Residual> &getResiduals() const { return residuals; }
  inline T getLogDepth(const Residual &res) {
    return parameters.logDepth(res.pointInd());
  }

  Hessian getHessian();
  Gradient getGradient();

  void precomputeValuesAndDerivatives();
  void clearPrecomputations();

  void optimize(int maxInterations);

private:
  using SecondFrameParametrization = SO3xS2Parametrization;
  using FrameParametrization = RightExpParametrization<SE3t>;

  class Parameters {
  public:
    static constexpr int sndDoF = SecondFrameParametrization::DoF;
    static constexpr int restDoF = FrameParametrization::DoF;
    static constexpr int affDoF = AffLightT::DoF;
    static constexpr int sndFrameDoF = sndDoF + affDoF;
    static constexpr int restFrameDoF = restDoF + affDoF;

    struct State {
      int frameParameters() const;
      void applyUpdate(const VecXt &deltaFrame, const VecXt &deltaPoints);

      State(KeyFrame **keyFrames, int newNumKeyFrames);

      VecXt logDepths;
      SE3t firstBodyToWorld;
      SecondFrameParametrization secondFrame;
      StdVector<FrameParametrization> restFrames;
      Array2d<AffLightT> lightWorldToFrame;
    };

    struct Jacobians {
      Jacobians(const State &state);
      Jacobians(const Parameters &parameters);

      SecondFrameParametrization::MatDiff dSecondFrame;
      StdVector<FrameParametrization::MatDiff> dRestFrames;
    };

    Parameters(CameraBundle *cam, KeyFrame **newKeyFrames, int newNumKeyFrames);

    SE3t getBodyToWorld(int frameInd) const;
    AffLightT getLightWorldToFrame(int frameInd, int frameCamInd) const;
    int numKeyFrames() const;
    int numPoints() const;
    int camBundleSize() const;
    int frameParameters() const;
    T logDepth(int i) const;

    void setPoints(std::vector<OptimizedPoint *> &&newOptimizedPoints);

    void update(const VecXt &deltaFrame, const VecXt &deltaPoints);
    State saveState() const;
    void recoverState(State oldState);
    void apply();

  private:
    State state;
    std::vector<OptimizedPoint *> optimizedPoints;
    std::vector<KeyFrame *> keyFrames;
  };

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
    T totalEnergy() const;

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

  Values createValues(PrecomputedHostToTarget &hostToTarget,
                      PrecomputedLightHostToTarget &lightHostToTarget);
  Values &getAllValues();
  Values &getAllValues(PrecomputedHostToTarget &hostToTarget,
                       PrecomputedLightHostToTarget &lightHostToTarget);
  Derivatives
  createDerivatives(const Values &values, PrecomputedHostToTarget &hostToTarget,
                    PrecomputedMotionDerivatives &motionDerivatives,
                    PrecomputedLightHostToTarget &lightHostToTarget);
  Derivatives &getDerivatives(PrecomputedHostToTarget &hostToTarget,
                              PrecomputedMotionDerivatives &motionDerivatives,
                              PrecomputedLightHostToTarget &lightHostToTarget);
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
