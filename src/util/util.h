#pragma once

#include <opencv2/core.hpp>

namespace fishdso {

// global image for debugging purposes
extern cv::Mat dbg;

void putDot(cv::Mat &img, cv::Point const &pos, cv::Scalar const &col);

void grad(cv::Mat const &img, cv::Mat &gradX, cv::Mat &gradY,
          cv::Mat &gradNorm);

} // namespace fishdso
