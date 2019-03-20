#ifndef INCLUDE_PIXELSELECTOR
#define INCLUDE_PIXELSELECTOR

#include "util/defs.h"
#include "util/settings.h"
#include <opencv2/opencv.hpp>

namespace fishdso {

class PixelSelector {
public:
  PixelSelector(const Settings::PixelSelector &settings = {});

  std::vector<cv::Point> select(const cv::Mat &frame, const cv::Mat1d &gradNorm,
                                int pointsNeeded, cv::Mat *debugOut);

private:
  std::vector<cv::Point> selectInternal(const cv::Mat &frame,
                                        const cv::Mat1d &gradNorm,
                                        int pointsNeeded, int blockSize,
                                        cv::Mat *debugOut);

  int lastBlockSize;
  int lastPointsFound;

  Settings::PixelSelector settings;
};

} // namespace fishdso

#endif
