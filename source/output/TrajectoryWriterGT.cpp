#include "output/TrajectoryWriterGT.h"

namespace mdso {

TrajectoryWriterGT::TrajectoryWriterGT(const SE3 _frameToWorldGT[],
                                       Timestamp _timestamps[], int size,
                                       const fs::path &outputDirectory,
                                       const fs::path &fileName)
    : frameToWorldGT(_frameToWorldGT, _frameToWorldGT + size)
    , timestamps(_timestamps, _timestamps + size)
    , mOutputFileName(outputDirectory / fileName) {}

void TrajectoryWriterGT::addToPool(const KeyFrame &keyFrame) {
  addToPoolByTimestamp(keyFrame.frames[0].timestamp);
}

void TrajectoryWriterGT::addToPool(const PreKeyFrame &frame) {
  addToPoolByTimestamp(frame.frames[0].timestamp);
}

void TrajectoryWriterGT::addToPoolByTimestamp(Timestamp ts) {
  auto tsIt = std::lower_bound(timestamps.begin(), timestamps.end(), ts);
  if (tsIt == timestamps.end() || *tsIt != ts)
    LOG(WARNING) << "Timestamp " << ts
                 << " not found in GT. Skipping this frame.";
  else
    frameToWorldGTPool.push({ts, frameToWorldGT[tsIt - timestamps.begin()]});
}

PosesPool &TrajectoryWriterGT::frameToWorldPool() { return frameToWorldGTPool; }

const fs::path &TrajectoryWriterGT::outputFileName() { return mOutputFileName; }

} // namespace mdso
