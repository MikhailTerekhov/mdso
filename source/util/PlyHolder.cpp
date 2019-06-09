#include "util/PlyHolder.h"
#include <glog/logging.h>

namespace fishdso {

const int countSpace = 19;

PlyHolder::PlyHolder(const std::string &fname)
    : fname(fname)
    , pointCount(0) {
  std::ofstream fs(fname);

  if (!fs.good())
    throw std::runtime_error("File \"" + fname + "\" could not be created.");

  fs << R"__(ply
format ascii 1.0
element vertex 0)__" +
            std::string(countSpace - 1, ' ') +
            R"__(
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
)__";

  fs.close();
}

void PlyHolder::putPoints(const std::vector<Vec3> &points,
                          const std::vector<cv::Vec3b> &colors) {
  LOG_IF(WARNING, points.size() != colors.size())
      << "Numbers of points and colors do not correspond in "
         "PlyHolder::putPoints."
      << std::endl;

  std::ofstream fs(fname, std::ios_base::app);
  int cnt = std::min(points.size(), colors.size());
  for (int i = 0; i < cnt; ++i) {
    const Vec3 &p = points[i];
    const cv::Vec3b &color = colors[i];
    fs << p[0] << ' ' << p[1] << ' ' << p[2] << ' ';
    fs << int(color[2]) << ' ' << int(color[1]) << ' ' << int(color[0]) << '\n';
  }

  pointCount += cnt;

  fs.close();
}

void PlyHolder::updatePointCount() {
  const int countPos = 36;
  std::fstream fs(fname);
  std::string emplacedValue = std::to_string(pointCount);
  emplacedValue += std::string(countSpace - emplacedValue.length(), ' ');
  fs.seekp(countPos) << emplacedValue;
  fs.close();
}

} // namespace fishdso
