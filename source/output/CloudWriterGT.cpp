#include "output/CloudWriterGT.h"

namespace fishdso {

CloudWriterGT::CloudWriterGT(
    const StdVector<SE3> &worldToFrameGT,
    const std::vector<std::vector<Vec3>> &pointsInFrameGT,
    const std::vector<std::vector<cv::Vec3b>> &colors,
    const std::string &outputDirectory, const std::string &fileName)
    : worldToFrameGT(worldToFrameGT)
    , pointsInFrameGT(pointsInFrameGT)
    , colors(colors)
    , cloudHolder(fileInDir(outputDirectory, fileName)) {}

void CloudWriterGT::initialized(
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

void CloudWriterGT::keyFramesMarginalized(
    const std::vector<const KeyFrame *> &marginalized) {
  CHECK(sim3Aligner);

  for (const KeyFrame *kf : marginalized) {
    std::vector<Vec3> points;
    points.reserve(pointsInFrameGT[kf->preKeyFrame->globalFrameNum].size());
    SE3 frameToWorld = kf->preKeyFrame->worldToThis.inverse();
    for (const Vec3 &p : pointsInFrameGT[kf->preKeyFrame->globalFrameNum])
      points.push_back(frameToWorld * sim3Aligner->alignScale(p));
    cloudHolder.putPoints(points, colors[kf->preKeyFrame->globalFrameNum]);

    for (const auto &preKeyFrame : kf->trackedFrames) {
      std::vector<Vec3> points;
      points.reserve(pointsInFrameGT[preKeyFrame->globalFrameNum].size());
      SE3 frameToWorld = preKeyFrame->worldToThis.inverse();
      for (const Vec3 &p : pointsInFrameGT[preKeyFrame->globalFrameNum])
        points.push_back(frameToWorld * sim3Aligner->alignScale(p));
      cloudHolder.putPoints(points, colors[preKeyFrame->globalFrameNum]);
    }
  }
  cloudHolder.updatePointCount();
}

void CloudWriterGT::destructed(
    const std::vector<const KeyFrame *> &lastKeyFrames) {
  keyFramesMarginalized(lastKeyFrames);
}

} // namespace fishdso
