#include "util/triangulation.h"
#include "util/types.h"
#include <chrono>
#include <opencv2/opencv.hpp>
#include <random>

using namespace fishdso;

int main() {
  const int segmentsCount = 10;
  const int onSegmCount = 5;
  stdvectorVec2 pnt;

  std::random_device rd;
  std::uniform_real_distribution<double> d(0, 100);
  std::uniform_real_distribution<double> d01(0, 1);

  for (int i = 0; i < segmentsCount; ++i) {
    Vec2 a(d(rd), d(rd)), b(d(rd), d(rd));
    for (int j = 0; j < onSegmCount; ++j) {
      double alpha = d01(rd);
      pnt.push_back(alpha * a + (1 - alpha) * b);
    }
  }

  Triangulation tri(pnt);
  cv::imshow("triang", tri.draw(800, 800));
  cv::waitKey();
  return 0;
}
