#pragma once

#include <opencv2/core.hpp>
#define _USE_MATH_DEFINES
#include <cmath>

namespace fishdso {

#define CV_RED cv::Scalar(0, 0, 255)
#define CV_GREEN cv::Scalar(0, 255, 0)
#define CV_BLUE cv::Scalar(255, 0, 0)
#define CV_BLACK cv::Scalar(0, 0, 0)

#define CV_BLACK_BYTE ((unsigned char)0)
#define CV_WHITE_BYTE ((unsigned char)255)

} // namespace fishdso
