#include "output/CloudWriterGT.h"

namespace mdso {

CloudWriterGT::CloudWriterGT(SE3 frameToWorldGT[], Timestamp timestamps[],
                             std::vector<Vec3> pointsInFrameGT[],
                             std::vector<cv::Vec3b> colors[], int size,
                             const fs::path &outputDirectory,
                             const fs::path &fileName)
    : timestamps(timestamps, timestamps + size)
    , frameToWorldGT(frameToWorldGT, frameToWorldGT + size)
    , pointsInFrameGT(pointsInFrameGT, pointsInFrameGT + size)
    , colors(colors, colors + size)
    , cloudHolder(outputDirectory / fileName) {}

int CloudWriterGT::findInd(Timestamp timestamp) {
  int ind = std::lower_bound(timestamps.begin(), timestamps.end(), timestamp) -
            timestamps.begin();
  CHECK(ind < timestamps.size() && timestamps[ind] == timestamp)
      << "timestamp not found in CloudWrierGT";
  return ind;
}

void CloudWriterGT::initialized(const KeyFrame *initializedKFs[], int size) {
  CHECK(size > 1);

  SE3 worldToFirst = initializedKFs[0]->thisToWorld.inverse();
  SE3 worldToLast = initializedKFs[size - 1]->thisToWorld.inverse();
  Timestamp firstTs = initializedKFs[0]->preKeyFrame->frames[0].timestamp;
  Timestamp lastTs = initializedKFs[size - 1]->preKeyFrame->frames[0].timestamp;

  SE3 worldToFirstGT = frameToWorldGT[findInd(firstTs)].inverse();
  SE3 worldToLastGT = frameToWorldGT[findInd(lastTs)].inverse();

  sim3Aligner = std::unique_ptr<Sim3Aligner>(new Sim3Aligner(
      worldToFirst, worldToLast, worldToFirstGT, worldToLastGT));

  for (auto &pose : frameToWorldGT)
    pose = sim3Aligner->alignWorldToFrameGT(pose).inverse();
}

void CloudWriterGT::keyFramesMarginalized(const KeyFrame *marginalized[],
                                          int size) {
  CHECK(sim3Aligner);

  for (int i = 0; i < size; ++i) {
    const KeyFrame *kf = marginalized[i];
    Timestamp ts = kf->preKeyFrame->frames[0].timestamp;
    int ind = findInd(ts);
    std::vector<Vec3> points;
    points.reserve(pointsInFrameGT[ind].size());
    for (const Vec3 &p : pointsInFrameGT[ind])
      points.push_back(kf->thisToWorld * sim3Aligner->alignScale(p));
    cloudHolder.putPoints(points, colors[ind]);

    for (const auto &preKeyFrame : kf->trackedFrames) {
      Timestamp pkfTs = kf->preKeyFrame->frames[0].timestamp;
      int pkfInd = findInd(pkfTs);
      std::vector<Vec3> points;
      points.reserve(pointsInFrameGT[pkfInd].size());
      SE3 frameToWorld = kf->thisToWorld * preKeyFrame->baseToThis.inverse();
      for (const Vec3 &p : pointsInFrameGT[pkfInd])
        points.push_back(frameToWorld * sim3Aligner->alignScale(p));
      cloudHolder.putPoints(points, colors[pkfInd]);
    }
  }
  cloudHolder.updatePointCount();
}

void CloudWriterGT::destructed(const KeyFrame *lastKeyFrames[], int size) {
  keyFramesMarginalized(lastKeyFrames, size);
}

} // namespace mdso
