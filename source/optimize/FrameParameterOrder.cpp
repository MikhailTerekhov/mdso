#include "optimize/FrameParameterOrder.h"

namespace mdso::optimize {

FrameParameterOrder::FrameParameterOrder(int numKeyFrames, int numCameras)
    : mNumKeyFrames(numKeyFrames)
    , mNumCameras(numCameras)
    , motionParams(sndDoF + (numKeyFrames - 2) * restDoF)
    , keyFrameAffStep(numCameras * affDoF) {}

int FrameParameterOrder::frameToWorld(int keyFrameInd) const {
  CHECK_GT(keyFrameInd, 0);
  CHECK_LT(keyFrameInd, mNumKeyFrames);

  if (keyFrameInd == 1)
    return 0;
  return sndDoF + (keyFrameInd - 2) * restDoF;
}

int FrameParameterOrder::lightWorldToFrame(int keyFrameInd, int camInd) const {
  CHECK_GT(keyFrameInd, 0);
  CHECK_LT(keyFrameInd, mNumKeyFrames);
  CHECK_GE(camInd, 0);
  CHECK_LT(camInd, mNumCameras);

  return motionParams + (keyFrameInd - 1) * keyFrameAffStep + camInd * affDoF;
}

int FrameParameterOrder::totalFrameParameters() const {
  return motionParams + (mNumKeyFrames - 1) * mNumCameras * affDoF;
}

} // namespace mdso::optimize