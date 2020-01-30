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

  EnergyFunction(CameraBundle *camBundle, KeyFrame *newKeyFrames[],
                 int numKeyFrames, const ResidualSettings &settings);

  inline const std::vector<OptimizedPoint *> &getOptimizedPoints() const {
    return points;
  }

  inline const std::vector<Residual> &getResiduals() const { return residuals; }

  Hessian getHessian();

private:
  using SecondFrameParametrization = SO3xS2Parametrization;
  using FrameParametrization = RightExpParametrization<SE3t>;

  struct OptimizationParams {
    OptimizationParams(CameraBundle *cam,
                       const std::vector<KeyFrame *> &keyFrames);

    SecondFrameParametrization secondFrame;
    StdVector<FrameParametrization> restFrames;
    Array2d<AffLightT> lightWorldToFrame;
    std::vector<T> logDepths;
  };

  SE3t getBodyToWorld(int frameInd) const;
  const MotionDerivatives &getHostToTargetDiff(int hostInd, int hostCamInd,
                                               int targetInd, int targetCamInd);

  AffLightT getLightWorldToFrame(int frameInd, int frameCamInd);
  const AffLightT &getLightHostToTarget(int hostInd, int hostCamInd,
                                        int targetInd, int targetCamInd);

  void recomputeHostToTarget();
  void resetLightHostToTarget();
  void resetPrecomputations();
  //  Array2d<AffLightT> computeLightBaseToTarget() const;

  std::unique_ptr<ceres::LossFunction> lossFunction;
  CameraBundle *cam;
  std::vector<Residual> residuals;
  std::vector<KeyFrame *> keyFrames;
  OptimizationParams optimizationParams;
  std::vector<OptimizedPoint *> points;
  Array4d<SE3t> hostToTarget;
  Array4d<std::optional<MotionDerivatives>> hostToTargetDiff;
  Array4d<std::optional<AffLightT>> lightHostToTarget;
  ResidualSettings settings;
};

} // namespace mdso::optimize

#endif
