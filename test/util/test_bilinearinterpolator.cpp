#include "util/BilinearInterpolator.h"
#include "util/types.h"
#include <ceres/cubic_interpolation.h>
#include <gtest/gtest.h>

using namespace mdso;

TEST(BilinearInterpolatorTest, ConstantFunction) {
  constexpr double step = 0.1;
  Mat22 vals = Mat22::Ones();
  ceres::Grid2D<double, 1, false> grid(vals.data(), 0, 2, 0, 2);
  BilinearInterpolator interpolator(grid);

  for (double r = 0; r <= 1; r += step)
    for (double c = 0; c <= 1; c += step) {
      double f, dfdr, dfdc;
      interpolator.Evaluate(r, c, &f, &dfdr, &dfdc);
      EXPECT_DOUBLE_EQ(f, 1);
      EXPECT_DOUBLE_EQ(dfdc, 0);
      EXPECT_DOUBLE_EQ(dfdr, 0);
    }
}

TEST(BilinearInterpolatorTest, GradientRow) {
  constexpr double step = 0.1;
  Mat22 vals;
  // clang-format off
  vals << 1, 1,
          3, 3;
  // clang-format on
  ceres::Grid2D<double, 1, false> grid(vals.data(), 0, 2, 0, 2);
  BilinearInterpolator interpolator(grid);

  for (double r = 0; r <= 1; r += step)
    for (double c = 0; c <= 1; c += step) {
      double f, dfdr, dfdc;
      interpolator.Evaluate(r, c, &f, &dfdr, &dfdc);
      EXPECT_DOUBLE_EQ(f, (1 - r) * 1 + r * 3);
      EXPECT_DOUBLE_EQ(dfdc, 0);
      EXPECT_DOUBLE_EQ(dfdr, 2);
    }
}

TEST(BilinearInterpolatorTest, GradientCol) {
  constexpr double step = 0.1;
  Mat22 vals;
  // clang-format off
  vals << 1, 3,
          1, 3;
  // clang-format on
  ceres::Grid2D<double, 1, false> grid(vals.data(), 0, 2, 0, 2);
  BilinearInterpolator interpolator(grid);

  for (double r = 0; r <= 1; r += step)
    for (double c = 0; c <= 1; c += step) {
      double f, dfdr, dfdc;
      interpolator.Evaluate(r, c, &f, &dfdr, &dfdc);
      EXPECT_DOUBLE_EQ(f, (1 - c) * 1 + c * 3);
      EXPECT_DOUBLE_EQ(dfdc, 2);
      EXPECT_DOUBLE_EQ(dfdr, 0);
    }
}

TEST(BilinearInterpolatorTest, GeneralUnoptimized) {
  constexpr double step = 0.1;
  constexpr double relErr = 1e-14;
  constexpr double f00 = 1, f01 = 2, f10 = 3, f11 = 4;
  Mat22 vals;
  // clang-format off
  vals << f00, f01,
          f10, f11;
  // clang-format on
  ceres::Grid2D<double, 1, false> grid(vals.data(), 0, 2, 0, 2);
  BilinearInterpolator interpolator(grid);

  for (double r = 0; r <= 1; r += step)
    for (double c = 0; c <= 1; c += step) {
      double f, dfdr, dfdc;
      interpolator.Evaluate(r, c, &f, &dfdr, &dfdc);
      EXPECT_NEAR(f,
                  (1 - r) * (1 - c) * f00 + (1 - r) * c * f01 +
                      r * (1 - c) * f10 + r * c * f11,
                  std::abs(f) * relErr);
      EXPECT_NEAR(dfdc, -(1 - r) * f00 + (1 - r) * f01 - r * f10 + r * f11,
                  std::abs(dfdc) * relErr);
      EXPECT_NEAR(dfdr, -(1 - c) * f00 - c * f01 + (1 - c) * f10 + c * f11,
                  std::abs(dfdr) * relErr);
    }
}

TEST(BilinearInterpolatorTest, ConstraintedValue) {
  constexpr int rows = 5, cols = 5;
  Eigen::Matrix<double, rows, cols> vals;
  std::mt19937 mt;
  std::uniform_real_distribution<double> d(0.01, 0.99);
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < cols; ++c)
      vals(r, c) = d(mt);

  ceres::Grid2D<double, 1, false> grid(vals.data(), 0, rows, 0, cols);
  BilinearInterpolator interpolator(grid);

  for (int r = 0; r < rows - 1; ++r)
    for (int c = 0; c < cols - 1; ++c) {
      double minv = std::min(
          {vals(r, c), vals(r, c + 1), vals(r + 1, c), vals(r + 1, c + 1)});
      double maxv = std::max(
          {vals(r, c), vals(r, c + 1), vals(r + 1, c), vals(r + 1, c + 1)});
      double f;
      double rr = r + d(mt), cc = c + d(mt);
      interpolator.Evaluate(rr, cc, &f);
      EXPECT_LE(f, maxv);
      EXPECT_GE(f, minv);
    }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
