#ifndef INCLUDE_DEFS
#define INCLUDE_DEFS

#include <cmath>
#include <opencv2/core.hpp>

namespace fishdso {

#define CV_RED cv::Scalar(0, 0, 255)
#define CV_GREEN cv::Scalar(0, 255, 0)
#define CV_BLUE cv::Scalar(255, 0, 0)
#define CV_MAGNETA cv::Scalar(255, 0, 255)
#define CV_BLACK cv::Scalar(0, 0, 0)

#define CV_BLACK_BYTE static_cast<unsigned char>(0)
#define CV_WHITE_BYTE static_cast<unsigned char>(255)

} // namespace fishdso

#endif
