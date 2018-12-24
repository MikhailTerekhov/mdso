#include "system/CameraModel.h"
#include "system/DsoSystem.h"
#include "util/geometry.h"
#include "util/types.h"
#include <Eigen/Core>
#include <gtest/gtest.h>
#include <tuple>

using namespace fishdso;

TEST(GeometryTest, IntersectOnSphereTest) {
  std::vector<std::tuple<double, Vec3, Vec3>> tests{
      {M_PI_2, Vec3(0, 0, 1), Vec3(1, 0, -1).normalized()},
      {M_PI_2, Vec3(1, 0, -1).normalized(), Vec3(0, 0, 1)},
      {M_PI_2, Vec3(1, 1, 0).normalized(), Vec3(1, -1, 0).normalized()},
      {M_PI_2, Vec3(1, 0, 1).normalized(), Vec3(-1, -1, -1).normalized()},
      {M_PI / 3, Vec3(0, 1, 0), Vec3(0, 0, 1)}};

  std::vector<std::pair<Vec3, Vec3>> answers{
      {Vec3(0, 0, 1), Vec3(1, 0, 0)},
      {Vec3(1, 0, 0), Vec3(0, 0, 1)},
      {Vec3(1, 1, 0).normalized(), Vec3(1, -1, 0).normalized()},
      {Vec3(1, 0, 1).normalized(), Vec3(0, -1, 0)},
      {Vec3(0, std::sqrt(3) / 2, 0.5), Vec3(0, 0, 1)}};

  std::vector<std::tuple<double, Vec3, Vec3>> testsFalse{
      {M_PI_2, Vec3(0, 0, -1), Vec3(-1, 0, -1).normalized()},
      {M_PI_2, Vec3(-1, 0, -1).normalized(), Vec3(0, -1, -1).normalized()},
      {M_PI * 0.75, Vec3(-1, 0, -100).normalized(),
       Vec3(0, -1, -100).normalized()}};

  const double eps = 1e-9;
  for (int i = 0; i < tests.size(); ++i) {
    ASSERT_TRUE(intersectOnSphere(std::get<0>(tests[i]), std::get<1>(tests[i]),
                                  std::get<2>(tests[i])))
        << "test #" << i << " failed: returned false" << std::endl;
    double err =
        std::sqrt((std::get<1>(tests[i]) - answers[i].first).squaredNorm() +
                  (std::get<2>(tests[i]) - answers[i].second).squaredNorm());
    ASSERT_LT(err, eps) << "test #" << i << " failed: error = " << err
                        << std::endl;
  }

  for (int i = 0; i < testsFalse.size(); ++i)
    ASSERT_FALSE(intersectOnSphere(std::get<0>(testsFalse[i]),
                                   std::get<1>(testsFalse[i]),
                                   std::get<2>(testsFalse[i])))
        << "false-test #" << i << " failed: returned true" << std::endl;
}

TEST(GeometryTest, TriangulateTest) {
  const int testCount = 1000;
  const double border = 1e2;
  const double eps = 1e-8;

  std::mt19937 mt;
  std::uniform_real_distribution<double> coord(-border, border);
  for (int it = 0; it < testCount; ++it) {
    Vec3 a(coord(mt), coord(mt), coord(mt));
    Vec3 b(coord(mt), coord(mt), coord(mt));
    Vec3 c(coord(mt), coord(mt), coord(mt));
    SO3 rot = SO3::sampleUniform(mt);

    double depth1 = (c - a).norm();
    double depth2 = (c - b).norm();

    Vec3 t = -(rot * (b - a));
    SE3 aToB(rot, t);

    Vec3 cDirInACoord = (c - a).normalized();
    Vec3 cDirInBCoord = (aToB * (c - a)).normalized();

    Vec2 result = triangulate(aToB, cDirInACoord, cDirInBCoord);
    Vec2 expected(depth1, depth2);
    double err = (result - expected).norm();
    ASSERT_LT(err, eps) << "test failed:\n"
                        << "a = " << a.transpose() << "\n"
                        << "b = " << b.transpose() << "\n"
                        << "c = " << c.transpose() << "\n"
                        << "rot = "
                        << rot.unit_quaternion().coeffs().transpose() << "\n"
                        << "err = " << err << "\n"
                        << "expected = " << expected.transpose() << "\n"
                        << "result   = " << result.transpose() << std::endl;
  }
}

TEST(GeometryTest, IsSameSideTest) {
  StdVector<std::array<Vec2, 4>> testsTrue{
      {Vec2(0, 0), Vec2(1, 0), Vec2(1, 1), Vec2(0, 1)},
      {Vec2(0, 0), Vec2(1, 1), Vec2(2, 2), Vec2(0, 1)},
      {Vec2(0, 0), Vec2(1, 1), Vec2(2, 2), Vec2(0, -1)},
      {Vec2(1, 1), Vec2(2, 1), Vec2(2, 1), Vec2(3, 1)}};
  StdVector<std::array<Vec2, 4>> testsFalse{
      {Vec2(0, 0), Vec2(1, 0), Vec2(1, 1), Vec2(1, -1)},
      {Vec2(1, 1), Vec2(1, -1), Vec2(-1, -1), Vec2(2, 2)},
      {Vec2(0, 0), Vec2(1, 0), Vec2(1e6, 1), Vec2(1e8, -1)},
      {Vec2(-1, -2), Vec2(0, 2), Vec2(-1, -1), Vec2(0, 0)}};

  for (int i = 0; i < testsTrue.size(); ++i)
    ASSERT_TRUE(isSameSide(testsTrue[i][0], testsTrue[i][1], testsTrue[i][2],
                           testsTrue[i][3]))
        << "test #" << i << " failed: returned false" << std::endl;
  for (int i = 0; i < testsFalse.size(); ++i)
    ASSERT_FALSE(isSameSide(testsFalse[i][0], testsFalse[i][1],
                            testsFalse[i][2], testsFalse[i][3]))
        << "test #" << i << " failed: returned true" << std::endl;
}

TEST(GeometryTest, InInsideTriangleTest) {
  StdVector<std::array<Vec2, 4>> testsTrue{
      {Vec2(0, 0), Vec2(3, 0), Vec2(0, 3), Vec2(1, 1)},
      {Vec2(0, 0), Vec2(1e4, 1), Vec2(2, 2), Vec2(1e4 - 1, 1)},
      {Vec2(0, 0), Vec2(2, 1), Vec2(0, 2), Vec2(0, 1)},
      {Vec2(0, 0), Vec2(2, 0), Vec2(2, 2), Vec2(2, 2)},
      {Vec2(0, 0), Vec2(1, 0), Vec2(2, 0), Vec2(1, 1)},
  };
  StdVector<std::array<Vec2, 4>> testsFalse{
      {Vec2(0, 0), Vec2(1, 0), Vec2(0, 1), Vec2(2, 0)},
      {Vec2(-1, -1), Vec2(0, 2), Vec2(2, 0), Vec2(-1, 1)},
      {Vec2(0, 0), Vec2(1e5, 1), Vec2(0, 1), Vec2(1, 0)}};

  for (int i = 0; i < testsTrue.size(); ++i)
    ASSERT_TRUE(isInsideTriangle(testsTrue[i][0], testsTrue[i][1],
                                 testsTrue[i][2], testsTrue[i][3]))
        << "test #" << i << " failed: returned false" << std::endl;
  for (int i = 0; i < testsFalse.size(); ++i)
    ASSERT_FALSE(isInsideTriangle(testsFalse[i][0], testsFalse[i][1],
                                  testsFalse[i][2], testsFalse[i][3]))
        << "test #" << i << " failed: returned true" << std::endl;
}

TEST(GeometryTest, IsABCDConvexTest) {
  StdVector<std::array<Vec2, 4>> testsTrue{
      {Vec2(0, 0), Vec2(1, 0), Vec2(1, 1), Vec2(0, 1)},
      {Vec2(-3, 0), Vec2(-2, 2), Vec2(3, 0), Vec2(2, -2)},
      {Vec2(0, 0), Vec2(0, 1), Vec2(0, 2), Vec2(1, 1)},
      {Vec2(-1e5, -1), Vec2(0, 0), Vec2(1e4, 1), Vec2(0, 1)}};
  StdVector<std::array<Vec2, 4>> testsFalse{
      {Vec2(0, 0), Vec2(1, 1), Vec2(2, 0), Vec2(1, 3)},
      {Vec2(-1e4, -1), Vec2(0, 0), Vec2(1e5, 1), Vec2(0, 1)},
      {Vec2(0, 0), Vec2(1, 1), Vec2(1, 0), Vec2(0, 1)}};

  for (int i = 0; i < testsTrue.size(); ++i)
    ASSERT_TRUE(isABCDConvex(testsTrue[i][0], testsTrue[i][1], testsTrue[i][2],
                             testsTrue[i][3]))
        << "test #" << i << " failed: returned false" << std::endl;
  for (int i = 0; i < testsFalse.size(); ++i)
    ASSERT_FALSE(isABCDConvex(testsFalse[i][0], testsFalse[i][1],
                              testsFalse[i][2], testsFalse[i][3]))
        << "test #" << i << " failed: returned true" << std::endl;
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
