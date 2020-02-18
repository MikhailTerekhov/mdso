#ifndef INCLUDE_FRAMEPARAMETERORDER
#define INCLUDE_FRAMEPARAMETERORDER

#include "optimize/parametrizations.h"
#include <glog/logging.h>

namespace mdso::optimize {

class FrameParameterOrder {
public:
  FrameParameterOrder(int numKeyFrames, int numCameras);

  int frameToWorld(int keyFrameInd) const;
  int lightWorldToFrame(int keyFrameInd, int camInd) const;
  int totalFrameParameters() const;
  inline int numKeyFrames() const { return mNumKeyFrames; }
  inline int numCameras() const { return mNumCameras; }

private:
  int mNumKeyFrames;
  int mNumCameras;
  int motionParams;
  int keyFrameAffStep;
};

} // namespace mdso::optimize

#endif
