#include "system/PreKeyFrame.h"
#include "PreKeyFrameInternals.h"
#include "system/KeyFrame.h"
#include "util/util.h"
#include <ceres/cubic_interpolation.h>
#include <opencv2/opencv.hpp>

namespace fishdso {

PreKeyFrame::FrameEntry::FrameEntry(const cv::Mat &_frameColored,
                                    const Settings::Pyramid &pyrSettings)
    : frameColored(_frameColored.clone())
    , framePyr(cvtBgrToGray(frameColored), pyrSettings.levelNum()) {
  grad(framePyr[0], gradX, gradY, gradNorm);
}

PreKeyFrame::PreKeyFrame(KeyFrame *baseFrame, CameraBundle *cam,
                         const cv::Mat coloredFrames[], int globalFrameNum,
                         long long timestamp,
                         const Settings::Pyramid &_pyrSettings)
    : baseFrame(baseFrame)
    , cam(cam)
    , globalFrameNum(globalFrameNum)
    , timestamp(timestamp)
    , pyrSettings(_pyrSettings) {
  for (int i = 0; i < frames.size(); ++i)
    frames.emplace_back(coloredFrames[i], pyrSettings);

  const ImagePyramid *refs[Settings::CameraBundle::max_camerasInBundle];
  for (int i = 0; i < frames.size(); ++i)
    refs[i] = &frames[i].framePyr;
  internals = std::unique_ptr<PreKeyFrameInternals>(
      new PreKeyFrameInternals(refs, frames.size(), pyrSettings));
}

PreKeyFrame::~PreKeyFrame() {}

}; // namespace fishdso
