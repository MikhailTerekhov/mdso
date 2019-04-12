#include "output/TrajectoryWriter.h"

namespace fishdso {

TrajectoryWriter::TrajectoryWriter(const std::string &outputDirectory,
                                   const std::string &fileName)
    : outputFileName(fileInDir(outputDirectory, fileName)) {}

void TrajectoryWriter::keyFramesMarginalized(
    const std::vector<const KeyFrame *> &marginalized) {
  std::ofstream posesOfs(outputFileName, std::ios_base::app);
  for (const KeyFrame *kf : marginalized) {
    SE3 worldToBase = kf->preKeyFrame->worldToThis;
    posesOfs << kf->preKeyFrame->globalFrameNum << ' ';
    putMotion(posesOfs, worldToBase);
    posesOfs << '\n';
    for (const auto &preKeyFrame : kf->trackedFrames) {
      posesOfs << preKeyFrame->globalFrameNum << ' ';
      putMotion(posesOfs, preKeyFrame->baseToThis * worldToBase);
      posesOfs << '\n';
    }
  }
}

void TrajectoryWriter::destructed(
    const std::vector<const KeyFrame *> &lastKeyFrames) {
  keyFramesMarginalized(lastKeyFrames);
}

} // namespace fishdso
