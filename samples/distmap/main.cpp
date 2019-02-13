#include "util/DistanceMap.h"
#include "util/defs.h"
#include "util/util.h"
#include <gflags/gflags.h>
#include <opencv2/opencv.hpp>

DEFINE_int32(w, 1920, "Width of the test image.");
DEFINE_int32(h, 1208, "Height of the test image.");
DEFINE_int32(init, 200, "Number of initial map points.");
DEFINE_int32(added, 200, "Number of points to choose from.");
DEFINE_int32(chosen, 50, "Number of points to choose.");

using namespace fishdso;

int main() {
  std::random_device rd;
  std::uniform_int_distribution<int> x(0, FLAGS_w - 1);
  std::uniform_int_distribution<int> y(0, FLAGS_h - 1);

  cv::Mat3b f(FLAGS_h, FLAGS_w, toCvVec3bDummy(CV_BLACK));
  StdVector<Vec2> init;
  init.reserve(FLAGS_init);
  for (int i = 0; i < FLAGS_init; ++i) {
    cv::Point p(x(rd) / 2, y(rd));
    init.push_back(toVec2(p));
    cv::circle(f, p, 6, CV_WHITE, cv::FILLED);
  }
  DistanceMap map(FLAGS_w, FLAGS_h, init);

  StdVector<Vec2> added;
  added.reserve(FLAGS_added);
  for (int i = 0; i < FLAGS_added; ++i) {
    cv::Point p((x(rd) + FLAGS_w) / 2, y(rd));
    added.push_back(toVec2(p));
    putCross(f, p, 4, CV_GREEN, 1);
  }
  std::vector<int> chosen = map.choose(added, FLAGS_chosen);
  for (int i : chosen)
    cv::circle(f, toCvPoint(added[i]), 6, CV_GREEN, 1);

  cv::imshow("frame", f);
  cv::waitKey();

  return 0;
}
