#include "system/CameraModel.h"
#include "system/DsoSystem.h"
#include "util/types.h"
#include <Eigen/Core>
#include <gtest/gtest.h>

using namespace mdso;

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
  EXPECT_LT(rmse, 0.1);
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
  EXPECT_LT(rmse, 0.0005);
}

TEST(CameraModelTest, CamerasPyramid) {
  double scale = 604.0;
  Vec2 center(1.58447, 1.07353);
  int unmapPolyDeg = 7;
  int pyrLevels = 6;
  VecX unmapPolyCoeffs(unmapPolyDeg, 1);
  unmapPolyCoeffs << 1.14544, -0.146714, -0.967996, 2.13329, -2.42001, 1.33018,
      -0.292722;
  int width = 1920, height = 1208;
  CameraModel cam(width, height, scale, center, unmapPolyCoeffs);
  StdVector<CameraModel> camPyr = cam.camPyr(pyrLevels);

  std::mt19937 mt;
  std::uniform_real_distribution<> xs(0, width - 1);
  std::uniform_real_distribution<> ys(0, height - 1);

  const int testCount = 1000;
  for (int lvl = 0; lvl < pyrLevels; ++lvl)
    for (int it = 0; it < testCount; ++it) {
      double x = xs(mt), y = ys(mt);
      Vec2 pnt(x, y);
      Vec2 pntScaled(x / (1 << lvl), y / (1 << lvl));
      Vec3 unmapOrig = cam.unmap(pnt).normalized();
      Vec3 unmapPyr = camPyr[lvl].unmap(pntScaled).normalized();
      double cos = unmapOrig.dot(unmapPyr);
      if (cos < -1)
        cos = -1;
      if (cos > 1)
        cos = 1;
      double angle = (180.0 / M_PI) * std::acos(cos);
      EXPECT_LT(angle, 0.01);
    }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
