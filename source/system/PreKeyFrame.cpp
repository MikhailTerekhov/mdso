#include "system/PreKeyFrame.h"
#include "util/util.h"
#include <opencv2/opencv.hpp>

namespace fishdso {

PreKeyFrame::PreKeyFrame(CameraModel *cam, const cv::Mat &frameColored,
                         int globalFrameNum)
    : frameColored(frameColored), framePyr(cvtBgrToGray(frameColored)),
      cam(cam), globalFrameNum(globalFrameNum) {}

}; // namespace fishdso
