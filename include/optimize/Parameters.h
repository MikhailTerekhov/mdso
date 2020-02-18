#ifndef INCLUDE_PARAMETERS
#define INCLUDE_PARAMETERS

#include "optimize/DeltaParameterVector.h"
#include "optimize/parametrizations.h"
#include "system/CameraBundle.h"

namespace mdso {

struct KeyFrame;
struct OptimizedPoint;

namespace optimize {

class Parameters {
public:
  struct State {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    State(KeyFrame **keyFrames, int newNumKeyFrames);

    int numKeyFrames() const;
    int numCameras() const;
    void applyUpdate(const DeltaParameterVector &delta);

    VecXt logDepths;
    SE3t firstBodyToWorld;
    SecondFrameParametrization secondFrame;
    StdVector<FrameParametrization> restFrames;
    Array2d<AffLightT> lightWorldToFrame;
  };

  struct Jacobians {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Jacobians(const State &state);
    Jacobians(const Parameters &parameters);

    SecondFrameParametrization::MatDiff dSecondFrame;
    StdVector<FrameParametrization::MatDiff> dRestFrames;
  };

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Parameters(CameraBundle *cam, KeyFrame **newKeyFrames, int newNumKeyFrames);

  SE3t getBodyToWorld(int frameInd) const;
  AffLightT getLightWorldToFrame(int frameInd, int frameCamInd) const;
  int numKeyFrames() const;
  int numPoints() const;
  int camBundleSize() const;
  T logDepth(int i) const;

  void setPoints(std::vector<OptimizedPoint *> &&newOptimizedPoints);

  void update(const DeltaParameterVector &delta);
  State saveState() const;
  void recoverState(State oldState);
  void apply();

private:
  State state;
  std::vector<OptimizedPoint *> optimizedPoints;
  std::vector<KeyFrame *> keyFrames;
};

} // namespace optimize
} // namespace mdso

#endif
