#include "system/KeyFrame.h"
#include "util/defs.h"
#include "util/settings.h"
#include "util/util.h"

namespace mdso {

KeyFrameEntry::KeyFrameEntry(const InitializedFrame::FrameEntry &entry,
                             KeyFrame *host, int ind,
                             const PointTracerSettings &tracingSettings)
    : host(host)
    , ind(ind)
    , timestamp(entry.timestamp)
    , preKeyFrameEntry(&host->preKeyFrame->frames[ind]) {
  immaturePoints.reserve(entry.depthedPoints.size());
  for (const auto &[p, d] : entry.depthedPoints) {
    if (preKeyFrameEntry->host->cam->bundle[ind].cam.isOnImage(
            p, tracingSettings.residualPattern.height)) {
      immaturePoints.emplace_back(this, p, tracingSettings);
      immaturePoints.back().setInitialDepth(d);
    }
  }
}

KeyFrameEntry::KeyFrameEntry(KeyFrame *host, int ind, Timestamp timestamp)
    : host(host)
    , ind(ind)
    , timestamp(timestamp)
    , preKeyFrameEntry(&host->preKeyFrame->frames[ind]) {}

KeyFrame::KeyFrame(const InitializedFrame &initializedFrame, CameraBundle *cam,
                   Preprocessor *preprocessor, int globalFrameNum,
                   PixelSelector pixelSelector[],
                   const Settings::KeyFrame &_kfSettings,
                   const Settings::Pyramid &pyrSettings,
                   const PointTracerSettings &tracingSettings)
    : thisToWorld(initializedFrame.thisToWorld)
    , kfSettings(_kfSettings) {
  std::vector<cv::Mat3b> images(cam->bundle.size());
  std::vector<Timestamp> timestamps(cam->bundle.size());
  for (int i = 0; i < cam->bundle.size(); ++i) {
    images[i] = initializedFrame.frames[i].frame;
    timestamps[i] = initializedFrame.frames[i].timestamp;
  }

  preKeyFrame = std::unique_ptr<PreKeyFrame>(
      new PreKeyFrame(this, cam, preprocessor, images.data(), globalFrameNum,
                      timestamps.data(), pyrSettings));

  frames.reserve(cam->bundle.size());
  for (int i = 0; i < cam->bundle.size(); ++i)
    frames.emplace_back(initializedFrame.frames[i], this, i, tracingSettings);
}

KeyFrame::KeyFrame(
    std::unique_ptr<PreKeyFrame> newPreKeyFrame,
    const std::vector<PixelSelector::PointVector> &newImmaturePoints,
    const Settings::KeyFrame &_kfSettings,
    const PointTracerSettings &tracingSettings)
    : preKeyFrame(std::move(newPreKeyFrame))
    , thisToWorld(preKeyFrame->baseFrame
                      ? preKeyFrame->baseFrame->thisToWorld() *
                            preKeyFrame->baseToThis().inverse()
                      : SE3())
    , kfSettings(_kfSettings) {
  int camNum = preKeyFrame->cam->bundle.size();
  frames.reserve(camNum);
  for (int i = 0; i < camNum; ++i) {
    frames.emplace_back(this, i, preKeyFrame->frames[i].timestamp);
    addImmatures(newImmaturePoints[i].data(), newImmaturePoints[i].size(), i,
                 &preKeyFrame->cam->bundle[i].cam, tracingSettings);
    frames[i].lightWorldToThis =
        preKeyFrame->baseFrame
            ? preKeyFrame->frames[i].lightBaseToThis *
                  preKeyFrame->baseFrame->frames[i].lightWorldToThis
            : preKeyFrame->frames[i].lightBaseToThis;
  }
}

std::vector<PixelSelector::PointVector> select(PreKeyFrame *preKeyFrame,
                                               PixelSelector pixelSelectors[],
                                               int totalPointsNeeded) {
  int numCameras = preKeyFrame->cam->bundle.size();
  std::vector<PixelSelector::PointVector> selected;
  selected.reserve(numCameras);
  for (int i = 0; i < numCameras; ++i)
    selected.push_back(pixelSelectors[i].select(
        preKeyFrame->frames[i].frameColored, preKeyFrame->frames[i].gradNorm,
        totalPointsNeeded / numCameras));
  return selected;
}

KeyFrame::KeyFrame(std::unique_ptr<PreKeyFrame> newPreKeyFrame,
                   PixelSelector pixelSelector[],
                   const Settings::KeyFrame &_kfSettings,
                   const PointTracerSettings &tracingSettings)
    : KeyFrame(std::move(newPreKeyFrame),
               select(newPreKeyFrame.get(), pixelSelector,
                      _kfSettings.immaturePointsNum()),
               _kfSettings, tracingSettings) {}

KeyFrame::KeyFrame(std::unique_ptr<PreKeyFrame> newPreKeyFrame,
                   const Settings::KeyFrame &_kfSettings,
                   const PointTracerSettings &tracingSettings)
    : KeyFrame(std::move(newPreKeyFrame),
               std::vector<PixelSelector::PointVector>(
                   newPreKeyFrame->cam->bundle.size()),
               _kfSettings, tracingSettings) {}

void KeyFrame::addImmatures(const cv::Point points[], int size, int numInBundle,
                            CameraModel *cam,
                            const PointTracerSettings &tracingSettings) {
  for (int i = 0; i < size; ++i) {
    Vec2 p = toVec2(points[i]);
    if (cam->isOnImage(p, tracingSettings.residualPattern.height))
      frames[numInBundle].immaturePoints.emplace_back(&frames[numInBundle], p,
                                                      tracingSettings);
  }
}

} // namespace mdso
