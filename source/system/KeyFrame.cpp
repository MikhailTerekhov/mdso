#include "system/KeyFrame.h"
#include "util/defs.h"
#include "util/settings.h"
#include "util/util.h"
#include <opencv2/opencv.hpp>

namespace mdso {

KeyFrameEntry::KeyFrameEntry(const InitializedFrame::FrameEntry &entry,
                             KeyFrame *host, int ind,
                             const PointTracerSettings &tracingSettings)
    : host(host)
    , ind(ind)
    , timestamp(entry.timestamp) {
  immaturePoints.reserve(entry.depthedPoints.size());
  for (const auto &[p, d] : entry.depthedPoints) {
    immaturePoints.emplace_back(this, p, tracingSettings);
    immaturePoints.back().depth = d;
  }
}

KeyFrameEntry::KeyFrameEntry(KeyFrame *host, int ind, Timestamp timestamp)

    : host(host)
    , ind(ind)
    , timestamp(timestamp) {}

KeyFrame::KeyFrame(const InitializedFrame &initializedFrame, CameraBundle *cam,
                   Preprocessor *preprocessor, int globalFrameNum,
                   PixelSelector pixelSelector[],
                   const Settings::KeyFrame &_kfSettings,
                   const Settings::Pyramid &pyrSettings,
                   const PointTracerSettings &tracingSettings)
    : thisToWorld(initializedFrame.thisToWorld)
    , kfSettings(_kfSettings) {
  std::vector<cv::Mat> images(cam->bundle.size());
  std::vector<Timestamp> timestamps(cam->bundle.size());
  for (int i = 0; i < cam->bundle.size(); ++i) {
    images[i] = initializedFrame.frames[i].frame;
    timestamps[i] = initializedFrame.frames[i].timestamp;
  }

  preKeyFrame = std::unique_ptr<PreKeyFrame>(
      new PreKeyFrame(this, cam, preprocessor, images.data(), globalFrameNum,
                      timestamps.data(), pyrSettings));

  for (int i = 0; i < cam->bundle.size(); ++i)
    frames.emplace_back(initializedFrame.frames[i], this, i, tracingSettings);
}

KeyFrame::KeyFrame(std::unique_ptr<PreKeyFrame> newPreKeyFrame,
                   PixelSelector pixelSelector[],
                   const Settings::KeyFrame &_kfSettings,
                   const PointTracerSettings &tracingSettings)
    : preKeyFrame(std::move(newPreKeyFrame))
    , thisToWorld(preKeyFrame->baseFrame
                      ? preKeyFrame->baseFrame->thisToWorld *
                            preKeyFrame->baseToThis().inverse()
                      : preKeyFrame->baseToThis().inverse())
    , kfSettings(_kfSettings) {
  int camNum = preKeyFrame->cam->bundle.size();
  for (int i = 0; i < camNum; ++i) {
    PixelSelector::PointVector selected = pixelSelector[i].select(
        preKeyFrame->image(i), preKeyFrame->frames[i].gradNorm,
        kfSettings.immaturePointsNum() / camNum, nullptr);
    frames.emplace_back(this, i, preKeyFrame->frames[i].timestamp);
    addImmatures(selected.data(), selected.size(), i, tracingSettings);
    frames[i].lightWorldToThis =
        preKeyFrame->baseFrame
            ? preKeyFrame->frames[i].lightBaseToThis *
                  preKeyFrame->baseFrame->frames[i].lightWorldToThis
            : preKeyFrame->frames[i].lightBaseToThis;
  }
}

void KeyFrame::addImmatures(const cv::Point points[], int size, int numInBundle,
                            const PointTracerSettings &tracingSettings) {
  for (int i = 0; i < size; ++i)
    frames[numInBundle].immaturePoints.emplace_back(
        &frames[numInBundle], toVec2(points[i]), tracingSettings);
}

} // namespace mdso
