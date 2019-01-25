#ifndef INCLUDE_PIXELSELECTOR
#define INCLUDE_PIXELSELECTOR

#include "util/settings.h"
#include <opencv2/opencv.hpp>

namespace fishdso {

class PixelSelector {
public:
  PixelSelector();

  std::vector<cv::Point> select(const cv::Mat &frame, const cv::Mat1d &gradNorm,
                                int pointsNeeded, cv::Mat *debugOut);

private:
  static constexpr int LI = settingInterestPointLayers;
  static constexpr int PL = settingPyrLevels;

  std::vector<cv::Point> selectInternal(const cv::Mat &frame,
                                        const cv::Mat1d &gradNorm,
                                        int pointsNeeded, int blockSize,
                                        cv::Mat *debugOut);

  int lastBlockSize;
  int lastPointsFound;
};

} // namespace fishdso

#endif
