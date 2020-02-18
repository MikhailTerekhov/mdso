#ifndef INCLUDE_GRADIENT
#define INCLUDE_GRADIENT

#include "optimize/Accumulator.h"
#include "optimize/DeltaParameterVector.h"
#include "optimize/Parameters.h"
#include "optimize/Residual.h"

namespace mdso::optimize {

class Gradient : public DeltaParameterVector {
public:
  class AccumulatedBlocks {
    friend class Gradient;

  public:
    AccumulatedBlocks(int numKeyFrames, int numCameras, int numPoints);

    int numKeyFrames() const;
    int numCameras() const;
    int numPoints() const;

    void add(const Residual &residual,
             const Residual::DeltaGradient &deltaGradient);

  private:
    void add(const Residual::FrameGradient &frameGradient, int fi, int ci);

    StdVector<Accumulator<Vec7t>> motion;
    Array2d<Accumulator<Vec2t>> aff;
    std::vector<Accumulator<T>> point;
  };

  Gradient(const AccumulatedBlocks &accumulatedBlocks,
           const Parameters::Jacobians &parameterJacobians);
};

} // namespace mdso::optimize

#endif
