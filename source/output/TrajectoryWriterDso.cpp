#include "output/TrajectoryWriterDso.h"

namespace mdso {

TrajectoryWriterDso::TrajectoryWriterDso(const fs::path &outputDirectory,
                                         const fs::path &fileName)
    : mOutputFileName(outputDirectory / fileName) {}

void TrajectoryWriterDso::addToPool(const KeyFrame &keyFrame) {
  mFrameToWorldPool.push(
      {keyFrame.preKeyFrame->frames[0].timestamp, keyFrame.thisToWorld()});
}

void TrajectoryWriterDso::addToPool(const PreKeyFrame &frame) {
  SE3 baseToWorld = frame.baseFrame->thisToWorld();
  SE3 thisToBase = frame.baseToThis().inverse();
  SE3 thisToWorld = baseToWorld * thisToBase;
  mFrameToWorldPool.push({frame.frames[0].timestamp, thisToWorld});
}

PosesPool &TrajectoryWriterDso::frameToWorldPool() { return mFrameToWorldPool; }

const fs::path &TrajectoryWriterDso::outputFileName() {
  return mOutputFileName;
}

} // namespace mdso