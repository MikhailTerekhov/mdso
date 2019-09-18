#include "output/TrajectoryWriterPredict.h"

namespace fishdso {

TrajectoryWriterPredict::TrajectoryWriterPredict(
    const fs::path &outputDirectory, const fs::path &fileName)
    : mOutputFileName(outputDirectory / fileName) {}

void TrajectoryWriterPredict::addToPool(const KeyFrame &keyFrame) {
  if (keyFrame.preKeyFrame->baseFrame == nullptr) {
    LOG(WARNING) << "nullptr base frame";
    mFrameToWorldPool.push(
        {keyFrame.frames[0].timestamp, keyFrame.thisToWorld});
  } else {
    // TODO base frame may not exist -- use std::weak_ptr
    // SE3 frameToWorldPredicted =
        // keyFrame.preKeyFrame->baseFrame->thisToWorld *
        // keyFrame.preKeyFrame->baseToThisPredicted.inverse();
    // mFrameToWorldPool.push(
        // {keyFrame.frames[0].timestamp, frameToWorldPredicted});
    mFrameToWorldPool.push(
        {keyFrame.frames[0].timestamp, keyFrame.thisToWorld});
  }
}

void TrajectoryWriterPredict::addToPool(const PreKeyFrame &frame) {
  SE3 frameToWorldPredicted =
      frame.baseFrame->thisToWorld * frame.baseToThisPredicted.inverse();
  mFrameToWorldPool.push({frame.frames[0].timestamp, frameToWorldPredicted});
}

PosesPool &TrajectoryWriterPredict::frameToWorldPool() {
  return mFrameToWorldPool;
}

const fs::path &TrajectoryWriterPredict::outputFileName() {
  return mOutputFileName;
}

} // namespace fishdso
