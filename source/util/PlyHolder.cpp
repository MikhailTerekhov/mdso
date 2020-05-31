#include "util/PlyHolder.h"
#include <fstream>
#include <glog/logging.h>

namespace mdso {

const int countSpace = 19;

PlyHolder::PlyHolder(const fs::path &fname, bool withStddevs)
    : fname(fname)
    , pointCount(0)
    , withStddevs(withStddevs) {
  std::ofstream fs(fname);

  CHECK(fs.good()) << "File \"" + fname.native() + "\" could not be created.";

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
)__";
  if (withStddevs)
    fs << "property float stddev\n";
  fs << "end_header\n";
}

void PlyHolder::putPoints(const std::vector<Vec3> &points,
                          const std::vector<cv::Vec3b> &colors) {
  CHECK(!withStddevs);
  putPoints(points, colors, nullptr);
}

void PlyHolder::putPoints(const std::vector<Vec3> &points,
                          const std::vector<cv::Vec3b> &colors,
                          const std::vector<double> &stddevs) {
  CHECK(withStddevs);
  putPoints(points, colors, &stddevs);
}

void PlyHolder::putPoints(const std::vector<Vec3> &points,
                          const std::vector<cv::Vec3b> &colors,
                          const std::vector<double> *stddevs) {
  CHECK_EQ(points.size(), colors.size());
  if (stddevs)
    CHECK_EQ(points.size(), stddevs->size());

  std::ofstream fs(fname, std::ios_base::app);
  int usedCount = 0;
  for (int i = 0; i < points.size(); ++i) {
    const Vec3 &p = points[i];
    bool isfinite =
        std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
    if (stddevs)
      isfinite = isfinite && std::isfinite((*stddevs)[i]);
    if (isfinite) {
      const cv::Vec3b &color = colors[i];
      fs << p[0] << ' ' << p[1] << ' ' << p[2] << ' ';
      fs << int(color[2]) << ' ' << int(color[1]) << ' ' << int(color[0]);
      if (stddevs)
        fs << ' ' << (*stddevs)[i];
      fs << '\n';
      usedCount++;
    }
  }

  pointCount += usedCount;

  fs.close();

  updatePointCount();
}

void PlyHolder::updatePointCount() {
  const int countPos = 36;
  std::fstream fs(fname);
  std::string emplacedValue = std::to_string(pointCount);
  emplacedValue += std::string(countSpace - emplacedValue.length(), ' ');
  fs.seekp(countPos) << emplacedValue;
  fs.close();
}

} // namespace mdso
