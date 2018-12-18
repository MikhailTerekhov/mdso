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
double minDepthCol = 0, maxDepthCol = 0;

double angle(const Vec3 &a, const Vec3 &b) {
  double cosAngle = a.normalized().dot(b.normalized());
  if (cosAngle < -1)
    cosAngle = -1;
  else if (cosAngle > 1)
    cosAngle = 1;
  return std::acos(cosAngle);
}

void setDepthColBounds(const std::vector<double> &depths) {
  std::vector<double> sorted = depths;
  std::sort(sorted.begin(), sorted.end());
  int redInd = FLAGS_red_depths_part * int(sorted.size());
  if (redInd < 0)
    redInd = 0;
  if (redInd >= sorted.size())
    redInd = sorted.size() - 1;

  int blueInd = FLAGS_blue_depths_part * sorted.size();
  if (blueInd < 0)
    blueInd = 0;
  if (blueInd >= sorted.size())
    blueInd = sorted.size() - 1;

  minDepthCol = sorted[redInd];
  maxDepthCol = sorted[blueInd];
}

void putMotion(std::ostream &out, const SE3 &motion) {
  out << motion.unit_quaternion().coeffs().transpose() << ' ';
  out << motion.translation().transpose();
}

void putDot(cv::Mat &img, const cv::Point &pos, const cv::Scalar &col) {
  cv::circle(img, pos, 4, col, cv::FILLED);
}

void putCross(cv::Mat &img, const cv::Point &pos, int size,
              const cv::Scalar &col, int thickness) {
  cv::line(img, pos - cv::Point(size, size), pos + cv::Point(size, size), col,
           thickness);
  cv::line(img, pos + cv::Point(-size, size), pos + cv::Point(size, -size), col,
           thickness);
}
void putSquare(cv::Mat &img, const cv::Point &pos, int size,
               const cv::Scalar &col, int thickness) {
  cv::rectangle(img, pos - cv::Point(size, size), pos + cv::Point(size, size),
                col, thickness);
}

void grad(const cv::Mat &img, cv::Mat1d &gradX, cv::Mat1d &gradY,
          cv::Mat1d &gradNorm) {
  static double filter[] = {-0.5, 0.0, 0.5};
  static cv::Mat1d gradXKer(1, 3, filter);
  static cv::Mat1d gradYKer(3, 1, filter);

  cv::filter2D(img, gradX, CV_64F, gradXKer, cv::Point(-1, -1), 0,
               cv::BORDER_REPLICATE);
  cv::filter2D(img, gradY, CV_64F, gradYKer, cv::Point(-1, -1), 0,
               cv::BORDER_REPLICATE);
  cv::magnitude(gradX, gradY, gradNorm);
}

double gradNormAt(const cv::Mat1b &img, const cv::Point &p) {
  double dx = (img(p.y, p.x + 1) - img(p.y, p.x - 1)) / 2.0;
  double dy = (img(p.y + 1, p.x) - img(p.y - 1, p.x)) / 2.0;
  return std::hypot(dx, dy);
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

void insertDepths(cv::Mat &img, const StdVector<Vec2> &points,
                  const std::vector<double> &depths, double minDepth,
                  double maxDepth, bool areSolidPnts) {
  if (points.size() != depths.size())
    throw std::runtime_error("insertDepths error!");
  if (points.empty())
    return;

  std::cout << "mind, maxd = " << minDepth << ' ' << maxDepth << std::endl;
  for (int i = 0; i < int(points.size()); ++i) {
    cv::Point cvp(static_cast<int>(points[i][0]),
                  static_cast<int>(points[i][1]));
    cv::circle(img, cvp, areSolidPnts ? 6 : 5,
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

template cv::Mat boxFilterPyrUp<unsigned char>(const cv::Mat &img);
template cv::Mat boxFilterPyrUp<cv::Vec3b>(const cv::Mat &img);

cv::Mat1b cvtBgrToGray(const cv::Mat &coloredImg) {
  cv::Mat result;
  cv::cvtColor(coloredImg, result, cv::COLOR_BGR2GRAY);
  return result;
}

cv::Mat pyrNUpDepth(const cv::Mat1d &integralWeightedDepths,
                    const cv::Mat1d &integralWeights, int levelNum) {
  cv::Mat1d res = cv::Mat1d((integralWeightedDepths.rows - 1) >> levelNum,
                            (integralWeightedDepths.cols - 1) >> levelNum);
  int d = (1 << levelNum);

  for (int y = 0; y < res.rows; ++y)
    for (int x = 0; x < res.cols; ++x) {
      int origX = x << levelNum, origY = y << levelNum;
      float depthsSum = integralWeightedDepths(origY + d, origX + d) -
                        integralWeightedDepths(origY, origX + d) -
                        integralWeightedDepths(origY + d, origX) +
                        integralWeightedDepths(origY, origX);
      float weightsSum = integralWeights(origY + d, origX + d) -
                         integralWeights(origY, origX + d) -
                         integralWeights(origY + d, origX) +
                         integralWeights(origY, origX);
      if (std::abs(weightsSum) > 1e-8)
        res(y, x) = depthsSum / weightsSum;
      else
        res(y, x) = -1;
    }
  return res;
}

cv::Mat drawDepthedFrame(const cv::Mat1b &frame, const cv::Mat1d &depths,
                         double minDepth, double maxDepth) {
  int w = frame.cols, h = frame.rows;
  cv::Mat3b res(h, w);
  cv::cvtColor(frame, res, cv::COLOR_GRAY2BGR);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      if (depths(y, x) > 0)
        res(y, x) = toCvVec3bDummy(depthCol(depths(y, x), minDepth, maxDepth));
  return res;
}

} // namespace fishdso
