#include "system/PreKeyFrame.h"
#include "util/util.h"
#include <opencv2/opencv.hpp>

namespace fishdso {

PreKeyFrame::PreKeyFrame(CameraModel *cam, const cv::Mat &frameColored,
                         int globalFrameNum)
    : framePyr(cvtBgrToGray(frameColored)),
      frameGrid(frame().data, 0, frame().rows, 0, frame().cols),
      frameInterpolator(frameGrid), cam(cam), globalFrameNum(globalFrameNum) {}

}; // namespace fishdso
