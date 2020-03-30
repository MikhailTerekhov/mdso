#include "system/PreKeyFrame.h"
#include "PreKeyFrameEntryInternals.h"
#include "system/FrameTracker.h"
#include "system/KeyFrame.h"
#include "util/util.h"
#include <ceres/cubic_interpolation.h>
#include <opencv2/opencv.hpp>

namespace mdso {

PreKeyFrame::FrameEntry::FrameEntry(PreKeyFrame *host, int ind,
                                    const cv::Mat3b &_frameColored,
                                    const cv::Mat1b &frameProcessed,
                                    Timestamp timestamp,
                                    const Settings::Pyramid &pyrSettings)
    : host(host)
    , ind(ind)
    , frameColored(_frameColored.clone())
    , framePyr(frameProcessed, pyrSettings.levelNum())
    , timestamp(timestamp)
    , internals(new PreKeyFrameEntryInternals(framePyr, pyrSettings)) {
  grad(framePyr[0], gradX, gradY, gradNorm);
}

PreKeyFrame::PreKeyFrame(KeyFrame *baseFrame, CameraBundle *cam,
                         const Preprocessor *preprocessor,
                         const cv::Mat3b coloredFrames[], int globalFrameNum,
                         Timestamp timestamps[],
                         const Settings::Pyramid &_pyrSettings)
    : baseFrame(baseFrame)
    , cam(cam)
    , globalFrameNum(globalFrameNum)
    , pyrSettings(_pyrSettings)
    , mWasTracked(false) {
  frames.reserve(cam->bundle.size());
  std::vector<cv::Mat1b> framesGray(cam->bundle.size()),
      framesProcessed(cam->bundle.size());
  for (int i = 0; i < cam->bundle.size(); ++i)
    framesGray[i] = cvtBgrToGray(coloredFrames[i]);

  preprocessor->process(framesGray.data(), framesProcessed.data(),
                        framesGray.size());

  for (int i = 0; i < cam->bundle.size(); ++i)
    frames.emplace_back(this, i, coloredFrames[i], framesProcessed[i],
                        timestamps[i], pyrSettings);
}

PreKeyFrame::~PreKeyFrame() {}

void PreKeyFrame::setTracked(const TrackingResult &trackingResult) {
  CHECK(!wasTracked());
  mBaseToThis = trackingResult.baseToTracked;
  for (int i = 0; i < cam->bundle.size(); ++i)
    frames[i].lightBaseToThis = trackingResult.lightBaseToTracked[i];
  mWasTracked = true;
}

}; // namespace mdso
