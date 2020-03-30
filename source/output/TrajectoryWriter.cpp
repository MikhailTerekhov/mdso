#include "output/TrajectoryWriter.h"

namespace mdso {

TrajectoryWriter::~TrajectoryWriter() {}

void TrajectoryWriter::newBaseFrame(const KeyFrame &baseFrame) {
  curKfTs.push_back(baseFrame.preKeyFrame->frames[0].timestamp);
}

void TrajectoryWriter::keyFramesMarginalized(const KeyFrame *marginalized[],
                                             int size) {
  std::vector<Timestamp> margKfTs;
  margKfTs.reserve(size);
  std::vector<Timestamp> newKfTs(curKfTs.size());

  for (int i = 0; i < size; ++i)
    margKfTs.push_back(marginalized[i]->preKeyFrame->frames[0].timestamp);

  int numLeft =
      std::set_difference(curKfTs.begin(), curKfTs.end(), margKfTs.begin(),
                          margKfTs.end(), newKfTs.begin()) -
      newKfTs.begin();
  newKfTs.resize(numLeft);
  curKfTs = std::move(newKfTs);

  for (int i = 0; i < size; ++i) {
    const KeyFrame *kf = marginalized[i];
    addToPool(*kf);
    for (const auto &preKeyFrame : kf->trackedFrames)
      addToPool(*preKeyFrame);
  }

  std::ofstream ofs(outputFileName(), std::ios_base::app);
  Timestamp minKfTs = curKfTs.empty() ? INF : curKfTs[0];
  PosesPool &pool = frameToWorldPool();
  while (!pool.empty() && pool.top().first < minKfTs) {
    const auto &top = pool.top();
    mWrittenFrameToWorld.push_back(top.second);
    putInMatrixForm(ofs, top.second);
    writtenKfTs.push_back(top.first);
    pool.pop();
  }
}

void TrajectoryWriter::destructed(const KeyFrame *lastKeyFrames[], int size) {
  keyFramesMarginalized(lastKeyFrames, size);
}

void TrajectoryWriter::saveTimestamps(const fs::path &timestampsFile) const {
  std::ofstream tsOfs(timestampsFile);
  for (Timestamp ts : writtenKfTs)
    tsOfs << ts << ' ';
}

} // namespace mdso
