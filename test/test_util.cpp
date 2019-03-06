#include "util/DepthedImagePyramid.h"
#include "util/PlyHolder.h"
#include "util/defs.h"
#include "util/settings.h"
#include "util/util.h"
#include <cstdio>
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

using namespace fishdso;

TEST(UtilTest, PyrNUpDepthTrivial) {
  for (int i = 0; i < 4; ++i) {
    cv::Mat1d w, d;
    w = cv::Mat1d(2, 2, 0.0);
    d = cv::Mat1d(2, 2, -1.0);
    w(i / 2, i % 2) = 1;
    d(i / 2, i % 2) = i + 1;
    cv::Mat1d intW;
    cv::integral(w, intW, CV_64F);
    cv::Mat1d wd = d.mul(w);
    cv::Mat1d intWD;
    cv::integral(wd, intWD, CV_64F);
    cv::Mat1d res = pyrNUpDepth(intWD, intW, 1);
    EXPECT_EQ(res.rows, 1);
    EXPECT_EQ(res.cols, 1);
    EXPECT_EQ(res(0, 0), i + 1);
  }

  for (int i = 0; i < 4; ++i) {
    cv::Mat1d w, d;
    w = cv::Mat1d(2, 2, 0.0);
    d = cv::Mat1d(2, 2, -1.0);
    w(i / 2, i % 2) = 1;
    d(i / 2, i % 2) = i + 1;
    cv::Mat1d intW;
    cv::integral(w, intW, CV_64F);
    cv::Mat1d wd = d.mul(w);
    cv::Mat1d intWD;
    cv::integral(wd, intWD, CV_64F);
    cv::Mat1d res = pyrNUpDepth(intWD, intW, 1);
    EXPECT_EQ(res.rows, 1);
    EXPECT_EQ(res.cols, 1);
    EXPECT_EQ(res(0, 0), i + 1);
  }
}

TEST(UtilTest, DepthedImagePyramid) {
  const int w = 1024, h = 512, cnt = 2000;

  std::mt19937 mt;
  std::uniform_int_distribution<int> x(0, w - 1), y(0, h - 1);
  std::uniform_real_distribution<double> ddis(10.0, 20.0);
  std::uniform_real_distribution<double> wdis(1.0, 2.0);
  StdVector<Vec2> pnts;
  std::vector<double> dps;
  std::vector<double> ws;
  for (int i = 0; i < cnt; ++i) {
    cv::Point p(x(mt), y(mt));
    pnts.push_back(toVec2(p));
    dps.push_back(ddis(mt));
    ws.push_back(wdis(mt));
  }

  cv::Mat1b base(h, w, CV_BLACK_BYTE);
  DepthedImagePyramid tst(base, pnts, dps, ws);

  for (int i = 0; i < pnts.size(); ++i) {
    for (int pl = 0; pl < settingPyrLevels; ++pl) {
      cv::Point p = toCvPoint(pnts[i]);
      ASSERT_GT(tst.depths[pl](p / (1 << pl)), 0)
          << "pl=" << pl << " p=" << p << " psh=" << p / (1 << pl) << " i=" << i
          << " d=" << dps[i] << " w=" << ws[i] << " porig=" << pnts[i]
          << std::endl;
    }
  }
}

TEST(UtilTest, PlyHolderTriv) {
  const int pntCount = 5;
  const std::string fname = "tst.ply";
  const std::string expected = R"__(ply
format ascii 1.0
element vertex 5                  )__"
                               R"__(
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
0 0 0 0 0 0
1 1 1 0 0 0
2 2 2 0 0 0
3 3 3 0 0 0
4 4 4 0 0 0
)__";
  std::vector<Vec3> points;
  std::vector<cv::Vec3b> colors;

  for (int i = 0; i < pntCount; ++i) {
    points.push_back(Vec3(i, i, i));
    colors.push_back(toCvVec3bDummy(CV_BLACK));
  }

  PlyHolder tester(fname);
  tester.putPoints(points, colors);
  tester.updatePointCount();
  std::ifstream resFs("tst.ply");
  std::stringstream ss;
  ss << resFs.rdbuf();
  EXPECT_EQ(ss.str(), expected);
  remove("tst.ply");
}

TEST(UtilTest, PlyHolderResize) {
  const int pntCount = 10;
  const std::string fname = "tst.ply";
  std::stringstream pntStr;
  for (int i = 0; i < pntCount; ++i)
    pntStr << i << ' ' << i + 1 << ' ' << i + 2 << " 255 0 0\n";
  const std::string expected = R"__(ply
format ascii 1.0
element vertex 10                 )__"
                               R"__(
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
)__" + pntStr.str();

  std::vector<Vec3> points;
  std::vector<cv::Vec3b> colors;

  PlyHolder tester(fname);
  for (int i = 0; i < pntCount / 2; ++i) {
    points.push_back(Vec3(i, i + 1, i + 2));
    colors.push_back(toCvVec3bDummy(CV_RED));
  }
  tester.putPoints(points, colors);
  tester.updatePointCount();
  points.clear();
  colors.clear();
  for (int i = pntCount / 2; i < pntCount; ++i) {
    points.push_back(Vec3(i, i + 1, i + 2));
    colors.push_back(toCvVec3bDummy(CV_RED));
  }
  tester.putPoints(points, colors);
  tester.updatePointCount();

  std::ifstream resFs("tst.ply");
  std::stringstream ss;
  ss << resFs.rdbuf();
  EXPECT_EQ(ss.str(), expected);
  remove("tst.ply");
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  // ::testing::GTEST_FLAG(filter) = "UtilTest.PlyHolderTriv";
  return RUN_ALL_TESTS();
}
