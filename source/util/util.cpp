#include "util/util.h"
#include "util/defs.h"
#include "util/settings.h"
#include <Eigen/Eigen>
#include <RelativePoseEstimator.h>
#include <algorithm>
#include <glog/logging.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include <sophus/sim3.hpp>
#include <utility>

bool validateDepthsPart(const char *flagname, double value) {
  if (value >= 0 && value <= 1)
    return true;
  std::cerr << "Invalid value for --" << std::string(flagname) << ": " << value
            << "\nit should be in [0, 1]" << std::endl;
  return false;
}

DEFINE_double(red_depths_part, 0,
              "Part of contrast points that will be drawn red (i.e. they are "
              "too close to be distinguished)");
DEFINE_validator(red_depths_part, validateDepthsPart);

DEFINE_double(blue_depths_part, 0.7,
              "Part of contrast points that will NOT be drawn completely blue "
              "(i.e. they are not too far to be distinguished)");
DEFINE_validator(blue_depths_part, validateDepthsPart);

namespace fishdso {

cv::Mat dbg;
double minDepthCol = 0, maxDepthCol = 1;

void printInPly(std::ostream &out, const std::vector<Vec3> &points,
                const std::vector<cv::Vec3b> &colors) {
  std::vector<Vec3> pf;
  std::vector<cv::Vec3b> cf;

  for (int i = 0; i < points.size(); ++i) {
    const Vec3 &p = points[i];
    const cv::Vec3b &color = colors[i];
    bool ok = true;
    for (int it = 0; it < 3; ++it)
      if (!std::isfinite(p[it]))
        ok = false;
    if (ok) {
      pf.push_back(p);
      cf.push_back(color);
    }
  }

  out.precision(15);
  out << R"__(ply
format ascii 1.0
element vertex )__"
      << pf.size() << R"__(
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
)__";

  for (int i = 0; i < pf.size(); ++i) {
    const Vec3 &p = pf[i];
    const cv::Vec3b &color = cf[i];

    out << p[0] << ' ' << p[1] << ' ' << p[2] << ' ';
    out << int(color[2]) << ' ' << int(color[1]) << ' ' << int(color[0])
        << '\n';
  }
}

void setDepthColBounds(const std::vector<double> &depths) {
  if (depths.empty())
    return;
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

cv::Mat drawLeveled(cv::Mat3b *images, int num, int w, int h, int resultW) {
  int downCnt = num / 2;
  int upCnt = num - downCnt;

  int upW = resultW / upCnt;
  int upH = double(h) / w * upW;
  int downW = resultW / downCnt;
  int downH = double(h) / w * downW;
  std::vector<cv::Mat> upRes(upCnt);
  std::vector<cv::Mat> downRes(downCnt);

  int pl = num - 1;
  for (; pl >= upCnt; --pl)
    cv::resize(images[pl], upRes[num - pl - 1], cv::Size(upW, upH), 0, 0,
               cv::INTER_NEAREST);
  for (; pl >= 0; --pl)
    cv::resize(images[pl], downRes[downCnt - pl - 1], cv::Size(downW, downH), 0,
               0, cv::INTER_NEAREST);
  cv::Mat upImg;
  cv::hconcat(upRes, upImg);
  cv::Mat downImg;
  cv::hconcat(downRes, downImg);
  cv::Mat result;
  cv::vconcat(upImg, downImg, result);
  return result;
}

void putMotion(std::ostream &out, const SE3 &motion) {
  out << motion.unit_quaternion().coeffs().transpose() << ' ';
  out << motion.translation().transpose();
}

void putInMatrixForm(std::ostream &out, const SE3 &motion) {
    Eigen::Matrix<double, 3, 4, Eigen::RowMajor> pose = motion.matrix3x4();
    for (int i = 0; i < 12; ++i)
      out << pose.data()[i] << ' ';
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

template cv::Mat boxFilterPyrDown<unsigned char>(const cv::Mat &img);
template cv::Mat boxFilterPyrDown<cv::Vec3b>(const cv::Mat &img);

cv::Mat1b cvtBgrToGray(const cv::Mat &coloredImg) {
  cv::Mat result;
  cv::cvtColor(coloredImg, result, cv::COLOR_BGR2GRAY);
  return result;
}

cv::Mat3b cvtBgrToGray3(const cv::Mat3b coloredImg) {
  cv::Mat1b result1C;
  cv::cvtColor(coloredImg, result1C, cv::COLOR_BGR2GRAY);
  cv::Mat3b result3C;
  cv::cvtColor(result1C, result3C, cv::COLOR_GRAY2BGR);
  return result3C;
}

cv::Mat3b cvtGrayToBgr(const cv::Mat1b &grayImg) {
  cv::Mat3b result;
  cv::cvtColor(grayImg, result, cv::COLOR_GRAY2BGR);
  return result;
}

cv::Mat1d pyrNUpDepth(const cv::Mat1d &integralWeightedDepths,
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

cv::Mat3b drawDepthedFrame(const cv::Mat1b &frame, const cv::Mat1d &depths,
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

std::string fileInDir(const std::string &directoryName,
                      const std::string &fileName) {
  return directoryName.size() > 1 && directoryName.back() == '/'
             ? directoryName + fileName
             : directoryName + "/" + fileName;
}

} // namespace fishdso
