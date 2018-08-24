#include "system/cameramodel.h"
#include "system/dsosystem.h"
#include "util/types.h"
#include <Eigen/Core>
#include <gtest/gtest.h>

using namespace fishdso;

TEST(CameraModelTest, CoreanCameraReprojection) {
  // some real-world data
  double scale = 604.0;
  Vec2 center(1.58492, 1.07424);
  int unmapPolyDeg = 5;
  VecX unmapPolyCoeffs(unmapPolyDeg, 1);
  unmapPolyCoeffs << 1.14169, -0.203229, -0.362134, 0.351011, -0.147191;
  int width = 1920, height = 1208;
  CameraModel cam(width, height, scale, center, unmapPolyCoeffs);

  std::srand(42);
  const int testnum = 2000;
  double sqErr = 0.0;

  for (int i = 0; i < testnum; ++i) {
    Vec2 pnt(double(rand() % width), double(rand() % height));
    Vec3 ray = cam.unmap(pnt.data());
    ray *= 10.5;

    Vec2 pntBack = cam.map(ray.data());
    sqErr += (pnt - pntBack).squaredNorm();
  }

  double rmse = std::sqrt(sqErr / testnum); // rmse in pixels
  std::cout << "reprojection rmse = " << rmse << std::endl;
  EXPECT_TRUE(rmse < 0.1);
}

TEST(CameraModelTest, SolelyPolynomial) {
  // here camera is initialised so that its mapping only inverses the given
  // polynomial
  double scale = 1;
  Vec2 center(0.0, 0.0);
  int unmapPolyDeg = 2;
  VecX unmapPolyCoefs(unmapPolyDeg, 1);
  unmapPolyCoefs << 1.0, -1.0;
  int width = 2, height = 0;
  CameraModel cam(width, height, scale, center, unmapPolyCoefs);

  std::srand(43);
  const int testnum = 2000;
  double sqErr = 0.0;
  for (int i = 0; i < testnum; ++i) {
    double x = double(std::rand()) / RAND_MAX;
    Vec3 pnt(x, 0, 1 - x * x);
    Vec2 projected = cam.map(pnt.data());
    sqErr += (projected - Vec2(x, 0)).squaredNorm();
  }
  double rmse = std::sqrt(sqErr / testnum);
  std::cout << "rmse = " << rmse << std::endl;
  EXPECT_TRUE(rmse < 0.0005);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
