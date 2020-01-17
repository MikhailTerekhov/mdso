#ifndef INCLUDE_PIXELSELECTOR
#define INCLUDE_PIXELSELECTOR

#include "util/defs.h"
#include "util/settings.h"
#include <opencv2/opencv.hpp>

namespace mdso {

class PixelSelector {
public:
  using PointVector = std::vector<cv::Point>;

  PixelSelector(const Settings::PixelSelector &settings = {});

  void initialize(const cv::Mat3b &frame, int pointsNeeded);
  PointVector select(const cv::Mat3b &frame, const cv::Mat1d &gradNorm,
                     int pointsNeeded, cv::Mat3b *debugOut = nullptr);

private:
  PointVector selectInternal(const cv::Mat3b &frame, const cv::Mat1d &gradNorm,
                             int pointsNeeded, int blockSize,
                             cv::Mat3b *debugOut);

  int lastBlockSize;
  int lastPointsFound;

  Settings::PixelSelector settings;

  std::mt19937 mt;
};

} // namespace mdso

#endif
