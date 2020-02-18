#include "optimize/Parameters.h"
#include "system/KeyFrame.h"

namespace mdso::optimize {

Parameters::Jacobians::Jacobians(const State &state)
    : dSecondFrame(state.secondFrame.diffPlus())
    , dRestFrames(state.restFrames.size()) {
  for (int i = 0; i < dRestFrames.size(); ++i)
    dRestFrames[i] = state.restFrames[i].diffPlus();
}

Parameters::Jacobians::Jacobians(const Parameters &parameters)
    : Jacobians(parameters.state) {}

Parameters::State::State(KeyFrame **keyFrames, int newNumKeyFrames)
    : firstBodyToWorld(keyFrames[0]->thisToWorld().cast<T>())
    , secondFrame(keyFrames[0]->thisToWorld(), keyFrames[1]->thisToWorld())
    , lightWorldToFrame(
          boost::extents[newNumKeyFrames]
                        [keyFrames[0]->preKeyFrame->cam->bundle.size()]) {
  CHECK_GE(newNumKeyFrames, 2);
  int bundleSize = keyFrames[0]->preKeyFrame->cam->bundle.size();

  restFrames.reserve(newNumKeyFrames - 2);
  for (int i = 2; i < newNumKeyFrames; ++i)
    restFrames.emplace_back(keyFrames[i]->thisToWorld().cast<T>());

  for (int fi = 0; fi < newNumKeyFrames; ++fi)
    for (int ci = 0; ci < bundleSize; ++ci)
      lightWorldToFrame[fi][ci] =
          keyFrames[fi]->frames[ci].lightWorldToThis.cast<T>();
}

int Parameters::State::numKeyFrames() const { return restFrames.size() + 2; }
int Parameters::State::numCameras() const {
  return lightWorldToFrame.shape()[1];
}

void Parameters::State::applyUpdate(const DeltaParameterVector &delta) {

  secondFrame.addDelta(delta.sndBlock());
  for (int fi = 2; fi < numKeyFrames(); ++fi)
    restFrames[fi - 2].addDelta(delta.restBlock(fi));

  for (int fi = 1; fi < numKeyFrames(); ++fi)
    for (int ci = 0; ci < numCameras(); ++ci)
      lightWorldToFrame[fi - 1][ci].applyUpdate(delta.affBlock(fi, ci));
}

Parameters::Parameters(CameraBundle *cam, KeyFrame **newKeyFrames,
                       int newNumKeyFrames)
    : state(newKeyFrames, newNumKeyFrames)
    , keyFrames(newKeyFrames, newKeyFrames + newNumKeyFrames) {}

SE3t Parameters::getBodyToWorld(int frameInd) const {
  CHECK_GE(frameInd, 0);
  CHECK_LT(frameInd, state.restFrames.size() + 2);
  if (frameInd == 0)
    return state.firstBodyToWorld;
  if (frameInd == 1)
    return state.secondFrame.value();
  return state.restFrames[frameInd - 2].value();
}

AffLightT Parameters::getLightWorldToFrame(int frameInd,
                                           int frameCamInd) const {
  return state.lightWorldToFrame[frameInd][frameCamInd];
}

int Parameters::numKeyFrames() const { return state.restFrames.size() + 2; }

int Parameters::numPoints() const { return state.logDepths.size(); }

int Parameters::camBundleSize() const {
  return state.lightWorldToFrame.shape()[1];
}

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
  CHECK_EQ(state.restFrames.size() + 2, keyFrames.size());
  CHECK_EQ(state.logDepths.size(), optimizedPoints.size());
}

void Parameters::update(const DeltaParameterVector &delta) {
  state.applyUpdate(delta);
}

void Parameters::apply() {
  keyFrames[1]->thisToWorld.setValue(state.secondFrame.value().cast<double>());
  for (int fi = 0; fi < state.restFrames.size(); ++fi)
    keyFrames[fi + 2]->thisToWorld.setValue(
        state.restFrames[fi].value().cast<double>());
  for (int pi = 0; pi < optimizedPoints.size(); ++pi)
    optimizedPoints[pi]->logDepth = state.logDepths[pi];
}

} // namespace mdso::optimize
