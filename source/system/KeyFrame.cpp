#include "system/KeyFrame.h"
#include "util/defs.h"
#include "util/settings.h"
#include "util/util.h"
#include <opencv2/opencv.hpp>

namespace fishdso {

KeyFrameEntry::KeyFrameEntry(const InitializedFrame::FrameEntry &entry,
                             KeyFrame *host, int ind)
    : host(host)
    , ind(ind)
    , timestamp(entry.timestamp) {
  immaturePoints.reserve(entry.depthedPoints.size());
  for (const auto &[p, d] : entry.depthedPoints) {
    immaturePoints.emplace_back(this, p, host->tracingSettings);
    immaturePoints.back().depth = d;
  }
}

KeyFrameEntry::KeyFrameEntry(KeyFrame *host, int ind, Timestamp timestamp)
    : host(host)
    , ind(ind)
    , timestamp(timestamp) {}

KeyFrame::KeyFrame(const InitializedFrame &initializedFrame, CameraBundle *cam,
                   int globalFrameNum, PixelSelector pixelSelector[],
                   const Settings::KeyFrame &_kfSettings,
                   const Settings::Pyramid &pyrSettings,
                   const PointTracerSettings &tracingSettings)
    : thisToWorld(initializedFrame.thisToWorld)
    , kfSettings(_kfSettings)
    , tracingSettings(tracingSettings) {
  std::vector<cv::Mat> images(cam->bundle.size());
  std::vector<Timestamp> timestamps(cam->bundle.size());
  for (int i = 0; i < cam->bundle.size(); ++i) {
    images[i] = initializedFrame.frames[i].frame;
    timestamps[i] = initializedFrame.frames[i].timestamp;
  }

  preKeyFrame = std::unique_ptr<PreKeyFrame>(
      new PreKeyFrame(this, cam, images.data(), globalFrameNum,
                      timestamps.data(), pyrSettings));

  for (int i = 0; i < cam->bundle.size(); ++i)
    frames.emplace_back(initializedFrame.frames[i], this, i);
}

KeyFrame::KeyFrame(std::unique_ptr<PreKeyFrame> newPreKeyFrame,
                   PixelSelector pixelSelector[],
                   const Settings::KeyFrame &_kfSettings,
                   const PointTracerSettings &tracingSettings)
    : preKeyFrame(std::move(newPreKeyFrame))
    , thisToWorld(preKeyFrame->baseFrame ? preKeyFrame->baseFrame->thisToWorld *
                                               preKeyFrame->baseToThis.inverse()
                                         : preKeyFrame->baseToThis.inverse())
    , kfSettings(_kfSettings)
    , tracingSettings(tracingSettings) {
  for (int i = 0; i < preKeyFrame->cam->bundle.size(); ++i) {
    PixelSelector::PointVector selected = pixelSelector[i].select(
        preKeyFrame->image(i), preKeyFrame->frames[i].gradNorm,
        kfSettings.immaturePointsNum(), nullptr);
    frames.emplace_back(this, i, preKeyFrame->frames[i].timestamp);
    addImmatures(selected.data(), selected.size(), i);
    frames[i].lightWorldToThis =
        preKeyFrame->baseFrame
            ? preKeyFrame->frames[i].lightBaseToThis *
                  preKeyFrame->baseFrame->frames[i].lightWorldToThis
            : preKeyFrame->frames[i].lightBaseToThis;
  }
}

void KeyFrame::addImmatures(const cv::Point points[], int size,
                            int numInBundle) {
  for (int i = 0; i < size; ++i)
    frames[numInBundle].immaturePoints.emplace_back(
        &frames[numInBundle], toVec2(points[i]), tracingSettings);
}

} // namespace fishdso
