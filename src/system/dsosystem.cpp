#include "dsosystem.h"
#include "../util/settings.h"

#ifdef DEBUG
#include <opencv2/highgui/highgui.hpp>
#endif

namespace fishdso {

DsoSystem::DsoSystem()
    : curFrameId(0), curPointId(0),
      adaptiveBlockSize(settingInitialAdaptiveBlockSize) {}

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
  cv::imshow("debug", keyframes.cbegin()->second->frameWithPoints);
}
#else
void DsoSystem::showDebug() {}
#endif

} // namespace fishdso
