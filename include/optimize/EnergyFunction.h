#ifndef INCLUDE_ENERGYFUNCTION
#define INCLUDE_ENERGYFUNCTION

#include "Residual.h"
#include "optimize/Parametrization.h"
#include "util/types.h"

namespace mdso::optimize {

class EnergyFunction {
public:
  struct Hessian {
    static constexpr int secondFramePars = 7;
    static constexpr int framePars = 8;

    MatXX frameFrame;
    MatXX framePoint;
    VecX pointPoint;
  };

  EnergyFunction(CameraBundle *camBundle, KeyFrame *keyFrames[], int size,
                 const ResidualSettings &settings);

  Hessian getHessian();

private:
  std::vector<Residual> residuals;

  KeyFrame *firstFrame;
  KeyFrame *secondFrame;
  SO3xS2Parametrization secondFrameParametrization;
  StdVector<std::pair<LeftExpParametrization<SE3t>, KeyFrame *>> frames;

  std::vector<std::pair<double, OptimizedPoint *>> points;

  ResidualSettings settings;
};

} // namespace mdso::optimize

#endif
