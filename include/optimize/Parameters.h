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
  class State {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    State(KeyFrame **keyFrames, int newNumKeyFrames);

    int numKeyFrames() const;
    int numCameras() const;

    const FrameParametrization &frameParametrization(int frameInd) const;
    FrameParametrization &frameParametrization(int frameInd);
    const AffLight &lightWorldToFrame(int frameInd, int frameCamInd) const;
    AffLight &lightWorldToFrame(int frameInd, int frameCamInd);

    void applyUpdate(const DeltaParameterVector &delta);

    VecXt logDepths;
    SE3t firstBodyToWorld;
    SecondFrameParametrization secondFrame;

  private:
    StdVector<FrameParametrization> restFrames;
    Array2d<AffLightT> mLightWorldToFrame;
  };

  class Jacobians {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Jacobians(const State &state);
    Jacobians(const Parameters &parameters);

    const SecondFrameParametrization::MatDiff &dSecondFrame() const;
    const FrameParametrization::MatDiff &dOtherFrame(int frameInd) const;

  private:
    SecondFrameParametrization::MatDiff mDSecondFrame;
    StdVector<FrameParametrization::MatDiff> mDRestFrames;
  };

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Parameters(CameraBundle *cam, KeyFrame **newKeyFrames, int newNumKeyFrames);

  SE3t getBodyToWorld(int frameInd) const;
  AffLightT getLightWorldToFrame(int frameInd, int frameCamInd) const;
  int numKeyFrames() const;
  int numPoints() const;
  int numCameras() const;
  T logDepth(int i) const;

  void setPoints(const std::vector<OptimizedPoint *> &newOptimizedPoints);

  void update(const DeltaParameterVector &delta);
  State saveState() const;
  State &stateRef() { return state; }
  void recoverState(State oldState);
  void apply() const;

private:
  State state;
  std::vector<OptimizedPoint *> optimizedPoints;
  std::vector<KeyFrame *> keyFrames;
};

} // namespace optimize

} // namespace mdso

#endif
