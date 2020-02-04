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

  VecRt getValues(int residualInd);
  inline const StdVector<Residual> &getResiduals() const { return residuals; }

  Hessian getHessian();
  Gradient getGradient();

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

    struct Jacobians {
      Jacobians(const Parameters &parameters);

      SecondFrameParametrization::MatDiff dSecondFrame;
      StdVector<FrameParametrization::MatDiff> dRestFrames;
    };

    Parameters(CameraBundle *cam, KeyFrame **keyFrames, int newNumKeyFrames);

    SE3t getBodyToWorld(int frameInd) const;
    AffLightT getLightWorldToFrame(int frameInd, int frameCamInd) const;
    int numKeyFrames() const;
    int numPoints() const;
    int camBundleSize() const;
    int frameParameters() const;

    void addPoint(T logDepth);
    const T &logDepth(int i) const;

  private:
    std::vector<T> logDepths;
    SE3t firstBodyToWorld;
    SecondFrameParametrization secondFrame;
    StdVector<FrameParametrization> restFrames;
    Array2d<AffLightT> lightWorldToFrame;
  };

  struct ValuesAndDerivatives {
    ValuesAndDerivatives(CameraBundle *cam, const Parameters &parameters,
                         const StdVector<Residual> &residuals);

    StdVector<VecRt> values;
    StdVector<Residual::Jacobian> residualJacobians;
    Parameters::Jacobians parametrizationJacobians;
  };

  class PrecomputedHostToTarget {
  public:
    PrecomputedHostToTarget(CameraBundle *cam, const Parameters *parameters);

    SE3t getHostToTarget(int hostInd, int hostCamInd, int targetInd,
                         int targetCamInd);
    const MotionDerivatives &getHostToTargetDiff(int hostInd, int hostCamInd,
                                                 int targetInd,
                                                 int targetCamInd);

  private:
    const Parameters *parameters;
    StdVector<SE3t> camToBody;
    StdVector<SE3t> bodyToCam;
    Array4d<SE3t> hostToTarget;
    Array4d<std::optional<MotionDerivatives>> hostToTargetDiff;
  };

  class PrecomputedLightHostToTarget {
  public:
    PrecomputedLightHostToTarget(const Parameters *parameters);

    AffLightT getLightHostToTarget(int hostInd, int hostCamInd, int targetInd,
                                   int targetCamInd);

  private:
    const Parameters *parameters;
    Array4d<std::optional<AffLightT>> lightHostToTarget;
  };

  StdVector<Residual> residuals;
  std::optional<ValuesAndDerivatives> valuesAndDerivatives;
  Parameters parameters;
  std::unique_ptr<ceres::LossFunction> lossFunction;
  CameraBundle *cam;
  ResidualSettings settings;
};

} // namespace mdso::optimize

#endif
