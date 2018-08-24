#include "system/dsosystem.h"
#include "util/settings.h"

#ifdef DEBUG
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#endif

namespace fishdso {

DsoSystem::DsoSystem(CameraModel *cam)
    : curFrameId(0), curPointId(0),
      adaptiveBlockSize(settingInitialAdaptiveBlockSize), cam(cam),
      dsoInitializer(std::make_unique<DsoInitializer>(this)),
      isInitialized(false) {}

void DsoSystem::addFrame(const cv::Mat &frame) {
  if (!isInitialized) {
    isInitialized = dsoInitializer->addFrame(frame);
  }
}

void DsoSystem::updateAdaptiveBlockSize(int curPointsDetected) {
  adaptiveBlockSize *= std::sqrt(static_cast<double>(curPointsDetected) /
                                 settingInterestPointsAdaptTo);
}

#ifdef DEBUG
void DsoSystem::showDebug() const {
  cv::Mat &lastKf = keyframes.cbegin()->second->frame;
  cv::Mat frameUndistort;
  Mat33 K;
  K << 800, 0, 850, 0, 600, 600, 0, 0, 1;
  cam->undistort<cv::Scalar>(lastKf, frameUndistort, K);
  cv::imshow("debug", frameUndistort);
}
#else
void DsoSystem::showDebug() const {}
#endif

} // namespace fishdso
