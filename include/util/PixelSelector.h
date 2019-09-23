#ifndef INCLUDE_PIXELSELECTOR
#define INCLUDE_PIXELSELECTOR

#include "util/defs.h"
#include "util/settings.h"
#include <opencv2/opencv.hpp>

namespace mdso {

class PixelSelector {
public:
  using PointVector =
      static_vector<cv::Point, Settings::PixelSelector::max_points>;

  PixelSelector(const Settings::PixelSelector &settings = {});

  PointVector select(const cv::Mat &frame, const cv::Mat1d &gradNorm,
                     int pointsNeeded, cv::Mat *debugOut = nullptr);

private:
  PointVector selectInternal(const cv::Mat &frame, const cv::Mat1d &gradNorm,
                             int pointsNeeded, int blockSize,
                             cv::Mat *debugOut);

  int lastBlockSize;
  int lastPointsFound;

  Settings::PixelSelector settings;

  std::mt19937 mt;
};

} // namespace mdso

#endif
