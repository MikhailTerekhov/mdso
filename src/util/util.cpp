#include "util/util.h"
#include "util/defs.h"
#include "util/settings.h"
#include <Eigen/Eigen>
#include <RelativePoseEstimator.h>
#include <algorithm>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include <sophus/sim3.hpp>
#include <utility>

namespace fishdso {

cv::Mat dbg;

void putDot(cv::Mat &img, cv::Point const &pos, cv::Scalar const &col) {
  cv::circle(img, pos, 4, col, cv::FILLED);
}

void grad(cv::Mat const &img, cv::Mat &gradX, cv::Mat &gradY,
          cv::Mat &gradNorm) {
  static float filter[] = {-1.0, 0.0, 1.0};
  static cv::Mat gradXKer(1, 3, CV_32FC1, filter);
  static cv::Mat gradYKer(3, 1, CV_32FC1, filter);

  cv::filter2D(img, gradX, CV_32F, gradXKer, cv::Point(-1, -1), 0,
               cv::BORDER_REPLICATE);
  cv::filter2D(img, gradY, CV_32F, gradYKer, cv::Point(-1, -1), 0,
               cv::BORDER_REPLICATE);
  cv::magnitude(gradX, gradY, gradNorm);
}

cv::Scalar depthCol(double d, double mind, double maxd) {
  if (d < mind)
    return CV_RED;
  if (d > maxd)
    return CV_BLUE;

  double mid = (mind + maxd) / 2;
  double dist = maxd - mid;
  return d > mid ? CV_GREEN * ((maxd - d) / dist) + CV_BLUE * ((d - mid) / dist)
                 : CV_RED * ((mid - d) / dist) + CV_GREEN * ((d - mind) / dist);
}

void insertDepths(cv::Mat &img, const std::vector<Vec2> &points,
                  const std::vector<double> &depths, double minDepth,
                  double maxDepth, bool areSolidPnts) {
  if (points.size() != depths.size())
    throw std::runtime_error("insertDepths error!");
  if (points.empty())
    return;
  //  int padding = int(double(depths.size()) * 0.05);

  //  std::vector<std::pair<Vec2, double>> pd(points.size());

  //  for (int i = 0; i < int(points.size()); ++i)
  //    pd[i] = {points[i], depths[i]};

  //  std::sort(pd.begin(), pd.end(),
  //            [](auto a, auto b) { return a.second < b.second; });

  //  double minDepth = pd[padding].second;
  //  double maxDepth = pd[int(pd.size()) - padding].second;

  std::cout << "mind, maxd = " << minDepth << ' ' << maxDepth << std::endl;
  for (int i = 0; i < int(points.size()); ++i) {
    cv::Point cvp(static_cast<int>(points[i][0]),
                  static_cast<int>(points[i][1]));
    cv::circle(img, cvp, areSolidPnts ? 6 : 4,
               depthCol(depths[i], minDepth, maxDepth),
               areSolidPnts ? cv::FILLED : 2);
    // putDot(result, cvp, depthCol(depths[i], minDepth, maxDepth));
  }
}

Vec2 toVec2(cv::Point p) { return Vec2(double(p.x), double(p.y)); }

cv::Vec3b toCvVec3bDummy(cv::Scalar scalar) {
  return cv::Vec3b(scalar[0], scalar[1], scalar[2]);
}

cv::Point toCvPoint(Vec2 vec) { return cv::Point(int(vec[0]), int(vec[1])); }

cv::Point toCvPoint(const Vec2 &vec, double scaleX, double scaleY,
                    cv::Point shift) {
  return cv::Point(int(vec[0] * scaleX) + shift.x,
                   int(vec[1] * scaleY) + shift.y);
}

} // namespace fishdso
