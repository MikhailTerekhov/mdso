#include "optimize/Gradient.h"

namespace mdso::optimize {

Gradient::AccumulatedBlocks::AccumulatedBlocks(int numKeyFrames, int numCameras,
                                               int numPoints)
    : motion(numKeyFrames - 1)
    , aff(boost::extents[numKeyFrames - 1][numCameras])
    , point(numPoints) {}

void Gradient::AccumulatedBlocks::add(
    const Residual::FrameGradient &frameGradient, int fi, int ci) {
  if (fi >= 0) {
    motion[fi] += frameGradient.qt;
    aff[fi][ci] += frameGradient.ab;
  }
}

void Gradient::AccumulatedBlocks::add(
    const Residual &residual, const Residual::DeltaGradient &deltaGradient) {
  int hi = residual.hostInd() - 1, hci = residual.hostCamInd(),
      ti = residual.targetInd() - 1, tci = residual.targetCamInd(),
      pi = residual.pointInd();
  add(deltaGradient.host, hi, hci);
  add(deltaGradient.target, ti, tci);

  point[pi] += deltaGradient.point;
}

int Gradient::AccumulatedBlocks::numKeyFrames() const {
  return motion.size() + 1;
}

int Gradient::AccumulatedBlocks::numCameras() const { return aff.shape()[1]; }

int Gradient::AccumulatedBlocks::numPoints() const { return point.size(); }

Gradient::Gradient(
    const mdso::optimize::Gradient::AccumulatedBlocks &accumulatedBlocks,
    const Parameters::Jacobians &parameterJacobians)
    : DeltaParameterVector(accumulatedBlocks.numKeyFrames(),
                           accumulatedBlocks.numCameras(),
                           accumulatedBlocks.numPoints()) {
  sndBlock() = parameterJacobians.dSecondFrame.transpose() *
               accumulatedBlocks.motion[0].accumulated();
  for (int fi = 2; fi < accumulatedBlocks.numKeyFrames(); ++fi)
    if (accumulatedBlocks.motion[fi - 1].wasUsed())
      restBlock(fi) = parameterJacobians.dRestFrames[fi - 2].transpose() *
                      accumulatedBlocks.motion[fi - 1].accumulated();

  for (int fi = 1; fi < accumulatedBlocks.numKeyFrames(); ++fi)
    for (int ci = 0; ci < accumulatedBlocks.numCameras(); ++ci)
      if (accumulatedBlocks.aff[fi - 1][ci].wasUsed())
        affBlock(fi, ci) = accumulatedBlocks.aff[fi - 1][ci].accumulated();

  for (int pi = 0; pi < accumulatedBlocks.numPoints(); ++pi)
    pointBlock(pi) = accumulatedBlocks.point[pi].accumulated();
}

} // namespace mdso::optimize