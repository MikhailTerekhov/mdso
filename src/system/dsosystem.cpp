#include "system/dsosystem.h"
#include "util/settings.h"

#ifdef DEBUG
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#endif

namespace fishdso {

DsoSystem::DsoSystem(CameraModel *cam)
    : dsoInitializer(cam), isInitialized(false) {}

void DsoSystem::addFrame(const cv::Mat &frame) {
  if (!isInitialized) {
    isInitialized = dsoInitializer.addFrame(frame);

    if (isInitialized) {
      std::vector<KeyFrame> kf =
          dsoInitializer.createKeyFrames(DsoInitializer::SPARSE_DEPTHS);
    }
  }
}

} // namespace fishdso
