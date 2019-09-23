#ifndef INCLUDE_PLYHOLDER
#define INCLUDE_PLYHOLDER

#include "util/types.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace mdso {

class PlyHolder {
public:
  PlyHolder(const fs::path &fname);

  void putPoints(const std::vector<Vec3> &points,
                 const std::vector<cv::Vec3b> &colors);
  void updatePointCount();

private:
  fs::path fname;
  int pointCount;
};

} // namespace mdso

#endif
