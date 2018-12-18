#include "util/Triangulation.h"
#include "util/defs.h"
#include "util/types.h"
#include <chrono>
#include <opencv2/opencv.hpp>
#include <random>

using namespace fishdso;

int main() {
  // const int segmentsCount = 10;
  // const int onSegmCount = 5;
  // StdVector<Vec2> pnt;

  std::random_device rd;
  std::uniform_real_distribution<double> d(0, 100);
  std::uniform_real_distribution<double> d01(0, 1);

  // for (int i = 0; i < segmentsCount; ++i) {
  // Vec2 a(d(rd), d(rd)), b(d(rd), d(rd));
  // for (int j = 0; j < onSegmCount; ++j) {
  // double alpha = d01(rd);
  // pnt.push_back(alpha * a + (1 - alpha) * b);
  // }
  // }

  const int pntCount = 100;
  StdVector<Vec2> pnt;
  for (int i = 0; i < pntCount; ++i) {
    pnt.push_back(Vec2(d(rd), d(rd)));
  }

  Triangulation tri(pnt);
  cv::Mat drawn = tri.draw(800, 800, CV_WHITE, CV_BLACK);
  cv::imshow("triang", drawn);
  cv::imwrite("triang.png", drawn);
  cv::waitKey();
  return 0;
}
