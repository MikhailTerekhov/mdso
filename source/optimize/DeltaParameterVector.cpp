#include "optimize/DeltaParameterVector.h"

namespace mdso::optimize {

DeltaParameterVector::DeltaParameterVector(int numKeyFrames, int numCameras,
                                           int numPoints)
    : frameParameterOrder(numKeyFrames, numCameras)
    , frame(frameParameterOrder.totalFrameParameters())
    , point(numPoints) {
  frame.setZero();
  point.setZero();
}

DeltaParameterVector::DeltaParameterVector(int numKeyFrames, int numCameras,
                                           const VecFt &frame,
                                           const VecPt &point)
    : frameParameterOrder(numKeyFrames, numCameras)
    , frame(frame)
    , point(point) {}

T DeltaParameterVector::dot(const DeltaParameterVector &other) const {
  return frame.dot(other.frame) + point.dot(other.point);
}

void DeltaParameterVector::setAffineZero() {
  for (int frameInd = 1; frameInd < frameParameterOrder.numKeyFrames();
       ++frameInd)
    for (int camInd = 0; camInd < frameParameterOrder.numCameras(); ++camInd)
      affBlock(frameInd, camInd).setZero();
}

} // namespace mdso::optimize