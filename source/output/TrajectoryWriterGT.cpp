#include "output/TrajectoryWriterGT.h"

namespace mdso {

TrajectoryWriterGT::TrajectoryWriterGT(const DatasetReader *datasetReader,
                                       const fs::path &outputDirectory,
                                       const fs::path &fileName)
    : datasetReader(datasetReader)
    , mOutputFileName(outputDirectory / fileName) {}

void TrajectoryWriterGT::addToPool(const KeyFrame &keyFrame) {
  addToPoolByTimestamp(keyFrame.frames[0].timestamp);
}

void TrajectoryWriterGT::addToPool(const PreKeyFrame &frame) {
  addToPoolByTimestamp(frame.frames[0].timestamp);
}

void TrajectoryWriterGT::addToPoolByTimestamp(Timestamp ts) {
  int frameInd = datasetReader->firstTimestampToInd(ts);

  if (!datasetReader->hasFrameToWorld(frameInd))
    LOG(WARNING) << "Timestamp " << ts
                 << " not found in GT. Skipping this frame.";
  else
    frameToWorldGTPool.push(
        {ts, datasetReader->frameToWorld(frameInd) * camToBody});
}

PosesPool &TrajectoryWriterGT::frameToWorldPool() { return frameToWorldGTPool; }

const fs::path &TrajectoryWriterGT::outputFileName() const {
  return mOutputFileName;
}

void TrajectoryWriterGT::setCamToBody(const SE3 &newCamToBody) {
  camToBody = newCamToBody;
}

} // namespace mdso
