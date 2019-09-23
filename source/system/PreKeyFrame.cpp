#include "system/PreKeyFrame.h"
#include "PreKeyFrameInternals.h"
#include "system/KeyFrame.h"
#include "util/util.h"
#include <ceres/cubic_interpolation.h>
#include <opencv2/opencv.hpp>

namespace fishdso {

PreKeyFrame::FrameEntry::FrameEntry(PreKeyFrame *host, int ind, const cv::Mat &_frameColored,
                                    Timestamp timestamp,
                                    const Settings::Pyramid &pyrSettings)
    : host(host)
    , ind(ind)
    , frameColored(_frameColored.clone())
    , framePyr(cvtBgrToGray(frameColored), pyrSettings.levelNum())
    , timestamp(timestamp) {
  grad(framePyr[0], gradX, gradY, gradNorm);
}

PreKeyFrame::PreKeyFrame(KeyFrame *baseFrame, CameraBundle *cam,
                         const cv::Mat coloredFrames[], int globalFrameNum,
                         Timestamp timestamps[],
                         const Settings::Pyramid &_pyrSettings)
    : baseFrame(baseFrame)
    , cam(cam)
    , globalFrameNum(globalFrameNum)
    , pyrSettings(_pyrSettings) {
  frames.reserve(cam->bundle.size());
  for (int i = 0; i < cam->bundle.size(); ++i)
    frames.emplace_back(this, i, coloredFrames[i], timestamps[i], pyrSettings);

  std::vector<const ImagePyramid *> refs(cam->bundle.size());
  for (int i = 0; i < frames.size(); ++i)
    refs[i] = &frames[i].framePyr;
  internals = std::unique_ptr<PreKeyFrameInternals>(
      new PreKeyFrameInternals(refs.data(), frames.size(), pyrSettings));
}

PreKeyFrame::~PreKeyFrame() {}

}; // namespace fishdso
