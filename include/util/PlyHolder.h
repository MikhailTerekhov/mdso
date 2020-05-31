#ifndef INCLUDE_PLYHOLDER
#define INCLUDE_PLYHOLDER

#include "util/types.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace mdso {

class PlyHolder {
public:
  PlyHolder(const fs::path &fname, bool withStddev = false);

  void putPoints(const std::vector<Vec3> &points,
                 const std::vector<cv::Vec3b> &colors);
  void putPoints(const std::vector<Vec3> &points,
                 const std::vector<cv::Vec3b> &colors,
                 const std::vector<double> &stddevs);
  void updatePointCount();

private:
  void putPoints(const std::vector<Vec3> &points,
                 const std::vector<cv::Vec3b> &colors,
                 const std::vector<double> *stddevs);

  fs::path fname;
  int pointCount;
  bool withStddevs;
};

} // namespace mdso

#endif
