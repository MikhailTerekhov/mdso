#include "optimize/Parameters.h"
#include "system/KeyFrame.h"

namespace mdso::optimize {

Parameters::State::State(KeyFrame **keyFrames, int newNumKeyFrames)
    : firstBodyToWorld(keyFrames[0]->thisToWorld().cast<T>())
    , secondFrame(keyFrames[0]->thisToWorld(), keyFrames[1]->thisToWorld())
    , mLightWorldToFrame(
          boost::extents[newNumKeyFrames]
                        [keyFrames[0]->preKeyFrame->cam->bundle.size()]) {
  CHECK_GE(newNumKeyFrames, 2);
  int bundleSize = keyFrames[0]->preKeyFrame->cam->bundle.size();

  restFrames.reserve(newNumKeyFrames - 2);
  for (int i = 2; i < newNumKeyFrames; ++i)
    restFrames.emplace_back(keyFrames[i]->thisToWorld().cast<T>());

  for (int fi = 0; fi < newNumKeyFrames; ++fi)
    for (int ci = 0; ci < bundleSize; ++ci)
      mLightWorldToFrame[fi][ci] =
          keyFrames[fi]->frames[ci].lightWorldToThis.cast<T>();
}

int Parameters::State::numKeyFrames() const { return restFrames.size() + 2; }

int Parameters::State::numCameras() const {
  return mLightWorldToFrame.shape()[1];
}

FrameParametrization &Parameters::State::frameParametrization(int frameInd) {
  return const_cast<FrameParametrization &>(
      const_cast<const Parameters::State *>(this)->frameParametrization(
          frameInd));
}

const FrameParametrization &
Parameters::State::frameParametrization(int frameInd) const {
  CHECK_GE(frameInd, 2);
  CHECK_LT(frameInd, numKeyFrames());
  return restFrames[frameInd - 2];
}

void Parameters::State::applyUpdate(const DeltaParameterVector &delta) {

  secondFrame.addDelta(delta.sndBlock());
  for (int fi = 2; fi < numKeyFrames(); ++fi)
    restFrames[fi - 2].addDelta(delta.restBlock(fi));

  for (int fi = 1; fi < numKeyFrames(); ++fi)
    for (int ci = 0; ci < numCameras(); ++ci)
      lightWorldToFrame(fi, ci).applyUpdate(delta.affBlock(fi, ci));
}

AffLight &Parameters::State::lightWorldToFrame(int frameInd, int frameCamInd) {
  // first frame affine light transformations should not be changed
  CHECK_GE(frameInd, 1);
  CHECK_LT(frameInd, numKeyFrames());
  CHECK_GE(frameCamInd, 0);
  CHECK_LT(frameCamInd, numCameras());
  return mLightWorldToFrame[frameInd][frameCamInd];
}

const AffLight &Parameters::State::lightWorldToFrame(int frameInd,
                                                     int frameCamInd) const {
  CHECK_GE(frameInd, 0);
  CHECK_LT(frameInd, numKeyFrames());
  CHECK_GE(frameCamInd, 0);
  CHECK_LT(frameCamInd, numCameras());
  return mLightWorldToFrame[frameInd][frameCamInd];
}

Parameters::Jacobians::Jacobians(const State &state)
    : mDSecondFrame(state.secondFrame.diffPlus())
    , mDRestFrames(state.numKeyFrames() - 2) {
  for (int fi = 2; fi < state.numKeyFrames(); ++fi)
    mDRestFrames[fi - 2] = state.frameParametrization(fi).diffPlus();
}

Parameters::Jacobians::Jacobians(const Parameters &parameters)
    : Jacobians(parameters.state) {}

const SecondFrameParametrization::MatDiff &
Parameters::Jacobians::dSecondFrame() const {
  return mDSecondFrame;
}

const FrameParametrization::MatDiff &
Parameters::Jacobians::dOtherFrame(int frameInd) const {
  CHECK_GE(frameInd, 2);
  CHECK_LT(frameInd, mDRestFrames.size() + 2);
  return mDRestFrames[frameInd - 2];
}

Parameters::Parameters(CameraBundle *cam, KeyFrame **newKeyFrames,
                       int newNumKeyFrames)
    : state(newKeyFrames, newNumKeyFrames)
    , keyFrames(newKeyFrames, newKeyFrames + newNumKeyFrames) {}

SE3t Parameters::getBodyToWorld(int frameInd) const {
  CHECK_GE(frameInd, 0);
  CHECK_LT(frameInd, state.numKeyFrames());
  if (frameInd == 0)
    return state.firstBodyToWorld;
  if (frameInd == 1)
    return state.secondFrame.value();
  return state.frameParametrization(frameInd).value();
}

AffLightT Parameters::getLightWorldToFrame(int frameInd,
                                           int frameCamInd) const {
  return state.lightWorldToFrame(frameInd, frameCamInd);
}

int Parameters::numKeyFrames() const { return state.numKeyFrames(); }

int Parameters::numPoints() const { return state.logDepths.size(); }

int Parameters::numCameras() const { return state.numCameras(); }

T Parameters::logDepth(int i) const {
  CHECK_GE(i, 0);
  CHECK_LT(i, state.logDepths.size());
  return state.logDepths[i];
}

void Parameters::setPoints(std::vector<OptimizedPoint *> &&newOptimizedPoints) {
  optimizedPoints = newOptimizedPoints;
  state.logDepths = VecXt(optimizedPoints.size());
  for (int i = 0; i < optimizedPoints.size(); ++i)
    state.logDepths[i] = optimizedPoints[i]->logDepth;
}

Parameters::State Parameters::saveState() const { return state; }

void Parameters::recoverState(State oldState) {
  state = std::move(oldState);
  CHECK_EQ(state.numKeyFrames(), keyFrames.size());
  CHECK_EQ(state.logDepths.size(), optimizedPoints.size());
}

void Parameters::update(const DeltaParameterVector &delta) {
  state.applyUpdate(delta);
}

void Parameters::apply() const {
  keyFrames[1]->thisToWorld.setValue(state.secondFrame.value().cast<double>());
  for (int fi = 2; fi < keyFrames.size(); ++fi)
    keyFrames[fi]->thisToWorld.setValue(
        state.frameParametrization(fi).value().cast<double>());

  for (int fi = 1; fi < keyFrames.size(); ++fi)
    for (int ci = 0; ci < numCameras(); ++ci) {
      keyFrames[fi]->frames[ci].lightWorldToThis =
          state.lightWorldToFrame(fi, ci).cast<double>();
      VLOG(1) << "kf #" << fi << ", cam #" << ci << ", new aff: \n"
              << keyFrames[fi]->frames[ci].lightWorldToThis;
    }

  for (int pi = 0; pi < optimizedPoints.size(); ++pi)
    optimizedPoints[pi]->logDepth = state.logDepths[pi];
}

} // namespace mdso::optimize
