#include "dsosystem.h"
#include "../util/settings.h"

#ifdef DEBUG
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#endif

namespace fishdso {

DsoSystem::DsoSystem(const CameraModel &cam)
    : curFrameId(0), curPointId(0),
      adaptiveBlockSize(settingInitialAdaptiveBlockSize), cam(cam) {}

void DsoSystem::addKf(cv::Mat frameColored) {
  keyframes[curFrameId] =
      std::make_unique<KeyFrame>(curFrameId, frameColored, this);
}

void DsoSystem::removeKf() {
  if (!keyframes.empty())
    keyframes.erase(keyframes.begin());
}

void DsoSystem::updateAdaptiveBlockSize(int curPointsDetected) {
  adaptiveBlockSize *=
      std::sqrt((double)curPointsDetected / settingInterestPointsAdaptTo);
}

#ifdef DEBUG
void DsoSystem::showDebug() {
  cv::Mat &lastKf = keyframes.cbegin()->second->frame;
  cv::Mat frameUndistort, smallUndistort;
  Mat33 K;
  K << 800, 0, 850, 0, 600, 600, 0, 0, 1;
  cam.undistort(lastKf, frameUndistort, K);
  cv::pyrDown(frameUndistort, smallUndistort,
              cv::Size(frameUndistort.cols / 2, frameUndistort.rows / 2));
  cv::imshow("debug", smallUndistort);
}
#else
void DsoSystem::showDebug() {}
#endif

} // namespace fishdso
