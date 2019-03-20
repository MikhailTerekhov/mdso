#include "system/PreKeyFrame.h"
#include "util/util.h"
#include <opencv2/opencv.hpp>

namespace fishdso {

PreKeyFrame::PreKeyFrame(CameraModel *cam, const cv::Mat &frameColored,
                         int globalFrameNum,
                         const Settings::Pyramid &_pyrSettings)
    : frameColored(frameColored)
    , framePyr(cvtBgrToGray(frameColored), _pyrSettings.levelNum)
    , cam(cam)
    , globalFrameNum(globalFrameNum)
    , pyrSettings(_pyrSettings) {
  grad(frame(), gradX, gradY, gradNorm);
}

}; // namespace fishdso
