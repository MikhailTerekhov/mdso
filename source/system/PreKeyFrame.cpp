#include "system/PreKeyFrame.h"
#include "PreKeyFrameInternals.h"
#include "system/KeyFrame.h"
#include "util/util.h"
#include <ceres/cubic_interpolation.h>
#include <opencv2/opencv.hpp>

namespace fishdso {

PreKeyFrame::FrameEntry::FrameEntry(const cv::Mat &_frameColored,
                                    long long timestamp,
                                    const Settings::Pyramid &pyrSettings)
    : frameColored(_frameColored.clone())
    , framePyr(cvtBgrToGray(frameColored), pyrSettings.levelNum())
    , timestamp(timestamp) {
  grad(framePyr[0], gradX, gradY, gradNorm);
}

PreKeyFrame::PreKeyFrame(KeyFrame *baseFrame, CameraBundle *cam,
                         const cv::Mat coloredFrames[], int globalFrameNum,
                         long long timestamps[],
                         const Settings::Pyramid &_pyrSettings)
    : baseFrame(baseFrame)
    , cam(cam)
    , globalFrameNum(globalFrameNum)
    , pyrSettings(_pyrSettings) {
  frames.reserve(cam->bundle.size());
  for (int i = 0; i < frames.size(); ++i)
    frames.emplace_back(coloredFrames[i], timestamps[i], pyrSettings);

  std::vector<const ImagePyramid *> refs(cam->bundle.size());
  for (int i = 0; i < frames.size(); ++i)
    refs[i] = &frames[i].framePyr;
  internals = std::unique_ptr<PreKeyFrameInternals>(
      new PreKeyFrameInternals(refs.data(), frames.size(), pyrSettings));
}

PreKeyFrame::~PreKeyFrame() {}

}; // namespace fishdso
