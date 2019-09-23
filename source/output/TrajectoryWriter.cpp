#include "output/TrajectoryWriter.h"

namespace fishdso {

TrajectoryWriter::TrajectoryWriter(const std::string &outputDirectory,
                                   const std::string &fileName)
    : outputFileName(fileInDir(outputDirectory, fileName)) {}

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
  std::swap(curKfTs, newKfTs);

  for (int i = 0; i < size; ++i) {
    const KeyFrame *kf = marginalized[i];
    SE3 baseToWorld = kf->thisToWorld;
    frameToWorldPool.push({kf->preKeyFrame->frames[0].timestamp, baseToWorld});
    for (const auto &preKeyFrame : kf->trackedFrames)
      frameToWorldPool.push({preKeyFrame->frames[0].timestamp,
                             baseToWorld * preKeyFrame->baseToThis.inverse()});
  }

  std::ofstream ofs(outputFileName, std::ios_base::app);
  int minKfTs = curKfTs.empty() ? INF : curKfTs[0];
  while (!frameToWorldPool.empty() && frameToWorldPool.top().first < minKfTs) {
    putInMatrixForm(ofs, frameToWorldPool.top().second);
    ofs << '\n';
    frameToWorldPool.pop();
  }
}

void TrajectoryWriter::destructed(const KeyFrame *lastKeyFrames[], int size) {
  keyFramesMarginalized(lastKeyFrames, size);
}

} // namespace fishdso
