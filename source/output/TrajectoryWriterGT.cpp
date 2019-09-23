#include "output/TrajectoryWriterGT.h"

namespace fishdso {

TrajectoryWriterGT::TrajectoryWriterGT(const SE3 frameToWorldGT[],
                                       Timestamp timestamps[], int size,
                                       const std::string &outputDirectory,
                                       const std::string &fileName)
    : outputFileName(fileInDir(outputDirectory, fileName)) {
  for (int i = 0; i < size; ++i)
    frameToWorldGTPool.push({timestamps[i], frameToWorldGT[i]});
}

void TrajectoryWriterGT::newBaseFrame(const KeyFrame &baseFrame) {
  curKfTs.push_back(baseFrame.preKeyFrame->frames[0].timestamp);
}

void TrajectoryWriterGT::keyFramesMarginalized(const KeyFrame *marginalized[],
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

  std::ofstream ofs(outputFileName, std::ios_base::app);
  int minKfTs = curKfTs.empty() ? INF : curKfTs[0];
  while (!frameToWorldGTPool.empty() &&
         frameToWorldGTPool.top().first < minKfTs) {
    putInMatrixForm(ofs, frameToWorldGTPool.top().second);
    ofs << '\n';
    frameToWorldGTPool.pop();
  }
}

void TrajectoryWriterGT::destructed(const KeyFrame *lastKeyFrames[], int size) {
  keyFramesMarginalized(lastKeyFrames, size);
}

} // namespace fishdso
