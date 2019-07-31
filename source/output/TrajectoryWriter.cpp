#include "output/TrajectoryWriter.h"

namespace fishdso {

TrajectoryWriter::TrajectoryWriter(const std::string &outputDirectory,
                                   const std::string &fileName,
                                   const std::string &matrixFormFileName)
    : outputFileName(fileInDir(outputDirectory, fileName))
    , matrixFormOutputFileName(fileInDir(outputDirectory, matrixFormFileName)) {
}

void TrajectoryWriter::newKeyFrame(const KeyFrame *keyFrame) {
  curKfNums.insert(keyFrame->preKeyFrame->globalFrameNum);
}

void TrajectoryWriter::keyFramesMarginalized(
    const std::vector<const KeyFrame *> &marginalized) {
  for (const KeyFrame *kf : marginalized)
    curKfNums.erase(kf->preKeyFrame->globalFrameNum);

  for (const KeyFrame *kf : marginalized) {
    SE3 baseToWorld = kf->thisToWorld;
    frameToWorldPool.insert({kf->preKeyFrame->globalFrameNum, baseToWorld});
    for (const auto &preKeyFrame : kf->trackedFrames)
      frameToWorldPool.insert(
          {preKeyFrame->globalFrameNum,
           baseToWorld * preKeyFrame->baseToThis.inverse()});
  }

  std::ofstream posesOfs(outputFileName, std::ios_base::app);
  std::ofstream matrixFormOfs(matrixFormOutputFileName, std::ios_base::app);
  int minKfNum = curKfNums.empty() ? INF : (*curKfNums.begin());
  auto it = frameToWorldPool.begin();
  while (it != frameToWorldPool.end() && it->first < minKfNum) {
    putInMatrixForm(matrixFormOfs, it->second);
    matrixFormOfs << '\n';

    posesOfs << it->first << ' ';
    putMotion(posesOfs, it->second.inverse());
    posesOfs << '\n';

    it = frameToWorldPool.erase(it);
  }
}

void TrajectoryWriter::destructed(
    const std::vector<const KeyFrame *> &lastKeyFrames) {
  keyFramesMarginalized(lastKeyFrames);
}

} // namespace fishdso
