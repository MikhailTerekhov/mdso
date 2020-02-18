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

} // namespace mdso::optimize