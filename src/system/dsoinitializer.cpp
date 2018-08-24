#include "system/dsoinitializer.h"
#include "util/defs.h"
#include "util/util.h"
#include <algorithm>
#include <opencv2/opencv.hpp>

namespace fishdso {

DsoInitializer::DsoInitializer(DsoSystem *dsoSystem)
    : dsoSystem(dsoSystem),
      stereoMatcher(std::make_unique<StereoMatcher>(dsoSystem->cam)),
      hasFirstFrame(false), framesSkipped(0) {}

bool DsoInitializer::addFrame(const cv::Mat &frame) {
  if (!hasFirstFrame) {
    addFirstFrame(frame);
    return false;
  } else {
    if (framesSkipped < settingFirstFramesSkip) {
      ++framesSkipped;
      return false;
    }

    stereoMatcher->createEstimations(frame);

    return true;
  }
}

void DsoInitializer::addFirstFrame(const cv::Mat &frame) {
  stereoMatcher->addBaseFrame(frame);
  hasFirstFrame = true;
}

} // namespace fishdso
