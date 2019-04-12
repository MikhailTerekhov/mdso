#ifndef INCLUDE_PLYHOLDER
#define INCLUDE_PLYHOLDER

#include "util/types.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace fishdso {

class PlyHolder {
public:
  PlyHolder(const std::string &fname);

  void putPoints(const std::vector<Vec3> &points,
                 const std::vector<cv::Vec3b> &colors);
  void updatePointCount();

private:
  std::string fname;
  int pointCount;
};

} // namespace fishdso

#endif
