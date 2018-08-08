#pragma once

#include <opencv2/core.hpp>

namespace fishdso {

class InterestPoint {
public:
  InterestPoint(cv::Point p);

private:
  int x, y;
};

} // namespace fishdso
