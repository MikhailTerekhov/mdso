#include "system/KeyFrame.h"
#include "util/defs.h"
#include "util/settings.h"
#include "util/util.h"
#include <opencv2/opencv.hpp>

namespace fishdso {

KeyFrame::KeyFrame(const InitializedFrame &initializedFrame, CameraBundle *cam,
                   int globalFrameNum, PixelSelector pixelSelector[],
                   const Settings::KeyFrame &_kfSettings,
                   const Settings::Pyramid &pyrSettings,
                   const PointTracerSettings &tracingSettings)
    : kfSettings(_kfSettings)
    , tracingSettings(tracingSettings) {
  cv::Mat frames[Settings::CameraBundle::max_camerasInBundle];
  long long timestamps[Settings::CameraBundle::max_camerasInBundle];
  for (int i = 0; i < cam->bundle.size(); ++i) {
    frames[i] = initializedFrame.frames[i].frame;
    timestamps[i] = initializedFrame.frames[i].timestamp;
  }

  preKeyFrame = std::unique_ptr<PreKeyFrame>(new PreKeyFrame(
      this, cam, frames, globalFrameNum, timestamps, pyrSettings));
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
        this, numInBundle, toVec2(points[i]), tracingSettings);
}

void KeyFrame::selectPointsDenser(PixelSelector pixelSelector[],
                                  int pointsNeeded) {
  for (int i = 0; i < preKeyFrame->cam->bundle.size(); ++i) {
    PixelSelector::PointVector points = pixelSelector[i].select(
        preKeyFrame->image(i), preKeyFrame->frames[i].gradNorm, pointsNeeded,
        nullptr);
    frames[i].immaturePoints.clear();
    frames[i].optimizedPoints.clear();
    addImmatures(points.data(), points.size(), i);
  }
}

} // namespace fishdso
