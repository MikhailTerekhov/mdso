#include "system/PreKeyFrame.h"
#include "util/util.h"
#include <opencv2/opencv.hpp>

namespace fishdso {

PreKeyFrame::PreKeyFrame(const cv::Mat &frameColored, int globalFrameNum)
    : areDepthsSet(false), globalFrameNum(globalFrameNum) {
  cv::cvtColor(frameColored, frame, cv::COLOR_BGR2GRAY);
}

}; // namespace fishdso
