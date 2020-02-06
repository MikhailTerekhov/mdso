#ifndef INCLUDE_ENERGYFUNCTION
#define INCLUDE_ENERGYFUNCTION

#include "Residual.h"
#include "optimize/parametrizations.h"
#include "util/types.h"
#include <optional>

namespace mdso::optimize {

class EnergyFunction {
public:
  struct Hessian {
    MatXXt frameFrame;
    MatXXt framePoint;
    VecXt pointPoint;
  };

  struct Gradient {
    VecXt frame;
    VecXt point;
  };

  EnergyFunction(CameraBundle *camBundle, KeyFrame *keyFrames[],
                 int numKeyFrames, const ResidualSettings &settings);

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

private:
  using Values = StdVector<VecRt>;
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
      void applyUpdate(const VecX &deltaFrame, const VecX &deltaPoints);

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

    void update(const VecX &deltaFrame, const VecX &deltaPoints);
    State saveState() const;
    void recoverState(State &&oldState);
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

  struct Derivatives {
    Derivatives(const Parameters &parameters,
                const StdVector<Residual> &residuals,
                PrecomputedHostToTarget &hostToTarget,
                PrecomputedMotionDerivatives &motionDerivatives,
                PrecomputedLightHostToTarget &lightHostToTarget);

    StdVector<Residual::Jacobian> residualJacobians;
    Parameters::Jacobians parametrizationJacobians;
  };

  Values &getAllValues();
  Values &getAllValues(PrecomputedHostToTarget &hostToTarget,
                       PrecomputedLightHostToTarget &lightHostToTarget);
  Derivatives &getDerivatives(PrecomputedHostToTarget &hostToTarget,
                              PrecomputedMotionDerivatives &motionDerivatives,
                              PrecomputedLightHostToTarget &lightHostToTarget);
  Hessian getHessian(PrecomputedHostToTarget &hostToTarget,
                     PrecomputedMotionDerivatives &motionDerivatives,
                     PrecomputedLightHostToTarget &lightHostToTarget);
  Gradient getGradient(PrecomputedHostToTarget &hostToTarget,
                       PrecomputedMotionDerivatives &motionDerivatives,
                       PrecomputedLightHostToTarget &lightHostToTarget);

  StdVector<Residual> residuals;
  std::optional<Values> values;
  std::optional<Derivatives> derivatives;
  Parameters parameters;
  std::unique_ptr<ceres::LossFunction> lossFunction;
  CameraBundle *cam;
  ResidualSettings settings;
};

} // namespace mdso::optimize

#endif
