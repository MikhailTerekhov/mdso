#include "util/triangulation.h"
#include "util/types.h"
#include <chrono>
#include <opencv2/opencv.hpp>
#include <random>

using namespace fishdso;

int main() {
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  const int segmentsCount = 10;
  const int onSegmCount = 5;
  const double width = 100, height = 100;
  std::vector<Vec2> pnt;

  std::mt19937 mt(seed);
  std::uniform_real_distribution<double> d(0, 100);
  std::uniform_real_distribution<double> d01(0, 1);

  for (int i = 0; i < segmentsCount; ++i) {
    Vec2 a(d(mt), d(mt)), b(d(mt), d(mt));
    for (int j = 0; j < onSegmCount; ++j) {
      double alpha = d01(mt);
      pnt.push_back(alpha * a + (1 - alpha) * b);
    }
  }

  Triangulation tri(pnt);
  cv::imshow("triang", tri.draw(800, 800));
  cv::waitKey();
  return 0;
}
