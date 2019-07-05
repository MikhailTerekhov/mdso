#include "output/TrajectoryWriterGT.h"

namespace fishdso {

TrajectoryWriterGT::TrajectoryWriterGT(
    const StdVector<SE3> &worldToFrameUnalignedGT,
    const std::string &outputDirectory, const std::string &fileName,
    const std::string &matrixFormFileName)
    : worldToFrameGT(worldToFrameUnalignedGT)
    , worldToFrameUnalignedGT(worldToFrameUnalignedGT)
    , outputFileName(fileInDir(outputDirectory, fileName))
    , matrixFormOutputFileName(fileInDir(outputDirectory, matrixFormFileName)) {
}

void TrajectoryWriterGT::initialized(
    const std::vector<const KeyFrame *> &initializedKFs) {
  CHECK(initializedKFs.size() > 1);

  SE3 worldToFirst = initializedKFs[0]->preKeyFrame->worldToThis;
  SE3 worldToLast = initializedKFs.back()->preKeyFrame->worldToThis;
  int firstNum = initializedKFs[0]->preKeyFrame->globalFrameNum;
  int lastNum = initializedKFs.back()->preKeyFrame->globalFrameNum;
  SE3 worldToFirstGT = worldToFrameGT[firstNum];
  SE3 worldToLastGT = worldToFrameGT[lastNum];

  sim3Aligner = std::unique_ptr<Sim3Aligner>(new Sim3Aligner(
      worldToFirst, worldToLast, worldToFirstGT, worldToLastGT));

  for (auto &pose : worldToFrameGT)
    pose = sim3Aligner->alignWorldToFrameGT(pose);
}

void TrajectoryWriterGT::keyFramesMarginalized(
    const std::vector<const KeyFrame *> &marginalized) {
  std::ofstream posesOfs(outputFileName, std::ios_base::app);
  std::ofstream matrixFormOfs(matrixFormOutputFileName, std::ios_base::app);

  for (const KeyFrame *kf : marginalized) {
    putInMatrixForm(
        matrixFormOfs,
        worldToFrameUnalignedGT[kf->preKeyFrame->globalFrameNum].inverse());
    matrixFormOfs << '\n';

    posesOfs << kf->preKeyFrame->globalFrameNum << ' ';
    putMotion(posesOfs, worldToFrameGT[kf->preKeyFrame->globalFrameNum]);
    posesOfs << '\n';
    for (const auto &preKeyFrame : kf->trackedFrames) {
      putInMatrixForm(
          matrixFormOfs,
          worldToFrameUnalignedGT[preKeyFrame->globalFrameNum].inverse());
      matrixFormOfs << '\n';

      posesOfs << preKeyFrame->globalFrameNum << ' ';
      putMotion(posesOfs, worldToFrameGT[preKeyFrame->globalFrameNum]);
      posesOfs << '\n';
    }
  }
}

void TrajectoryWriterGT::destructed(
    const std::vector<const KeyFrame *> &lastKeyFrames) {
  keyFramesMarginalized(lastKeyFrames);
}

} // namespace fishdso
