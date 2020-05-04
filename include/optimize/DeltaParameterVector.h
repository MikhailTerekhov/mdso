#ifndef INCLUDE_DELTAPARAMETERVECTOR
#define INCLUDE_DELTAPARAMETERVECTOR

#include "optimize/FrameParameterOrder.h"
#include "optimize/parametrizations.h"

namespace mdso::optimize {

class DeltaParameterVector {
public:
  using VecFt = VecXt;
  using VecPt = VecXt;

  static constexpr int sndDoF = SecondFrameParametrization::DoF;
  static constexpr int restDoF = FrameParametrization::DoF;
  static constexpr int affDoF = AffLightT::DoF;

  DeltaParameterVector(int numKeyFrames, int numCameras, int numPoints);
  DeltaParameterVector(int numKeyFrames, int numCameras, const VecFt &frame,
                       const VecPt &point);

  inline T &pointBlock(int pointInd) { return point[pointInd]; }
  inline const T &pointBlock(int pointInd) const { return point[pointInd]; }

  inline const VecXt &getFrame() const { return frame; }
  inline const VecXt &getPoint() const { return point; }

  friend DeltaParameterVector operator*(T factor,
                                        const DeltaParameterVector &delta);
  T dot(const DeltaParameterVector &other) const;

  void setAffineZero();
  void constraintDepths(double maxAbsDeltaD);

  inline Eigen::Block<VecFt, sndDoF, 1> sndBlock() {
    return frame.head<sndDoF>();
  }
  inline Eigen::Block<const VecFt, sndDoF, 1> sndBlock() const {
    return frame.head<sndDoF>();
  }
  inline Eigen::Block<VecFt, restDoF, 1> restBlock(int frameInd) {
    return frame.segment<restDoF>(frameParameterOrder.frameToWorld(frameInd));
  }
  inline Eigen::Block<const VecFt, restDoF, 1> restBlock(int frameInd) const {
    return frame.segment<restDoF>(frameParameterOrder.frameToWorld(frameInd));
  }
  inline Eigen::Block<VecFt, affDoF, 1> affBlock(int frameInd,
                                                 int frameCamInd) {
    return frame.segment<affDoF>(
        frameParameterOrder.lightWorldToFrame(frameInd, frameCamInd));
  }
  inline Eigen::Block<const VecFt, affDoF, 1> affBlock(int frameInd,
                                                       int frameCamInd) const {
    return frame.segment<affDoF>(
        frameParameterOrder.lightWorldToFrame(frameInd, frameCamInd));
  }

private:
  FrameParameterOrder frameParameterOrder;
  VecFt frame;
  VecPt point;
};

} // namespace mdso::optimize

#endif
