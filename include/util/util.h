#ifndef INCLUDE_UTIL
#define INCLUDE_UTIL

#include "util/settings.h"
#include "util/types.h"
#include <opencv2/opencv.hpp>
#include <vector>

DECLARE_double(red_depths_part);
DECLARE_double(blue_depths_part);

namespace mdso {

extern cv::Mat dbg;
extern double minDepthCol, maxDepthCol;

template <typename T> inline std::vector<T> reservedVector(int toReserve) {
  std::vector<T> res;
  res.reserve(toReserve);
  return res;
}

template <typename T>
void outputArrayUndivided(std::ostream &os, const T array[], int size) {
  std::stringstream ss;
  for (int i = 0; i < size; ++i)
    ss << array[i] << ' ';
  os << ss.str();
}

template <typename T>
void outputArray(const std::string &fname, const T array[], int size) {
  std::ofstream ofs(fname);
  for (int i = 0; i < size; ++i)
    ofs << array[i] << ' ';
  ofs << std::endl;
  ofs.close();
}

template <typename T>
void outputArray(const std::string &fname, const std::vector<T> &array) {
  outputArray(fname, array.data(), array.size());
}

void printInPly(std::ostream &out, const std::vector<Vec3> &points,
                const std::vector<cv::Vec3b> &colors);

void setDepthColBounds(const std::vector<double> &depths);

cv::Mat3b drawLeveled(cv::Mat3b *images, int num, int w, int h, int resutW);

void putMotion(std::ostream &out, const SE3 &motion);
void putInMatrixForm(std::ostream &out, const SE3 &motion);

void putDot(cv::Mat &img, const cv::Point &pos, const cv::Scalar &col);
void putCross(cv::Mat &img, const cv::Point &pos, int size,
              const cv::Scalar &col, int thickness);
void putSquare(cv::Mat &img, const cv::Point &pos, int size,
               const cv::Scalar &col, int thickness);

void grad(const cv::Mat1b &img, cv::Mat1d &gradX, cv::Mat1d &gradY,
          cv::Mat1d &gradNorm);
double gradNormAt(const cv::Mat1b &img, const cv::Point &p);

cv::Scalar depthCol(double d, double mind, double maxd);

void insertDepths(cv::Mat &img, const Vec2 points[], const double depths[],
                  int size, double minDepth, double maxDepth,
                  bool areSolidPnts);

Vec2 toVec2(cv::Point p);
Vec2i toVec2i(cv::Point p);
cv::Point toCvPoint(Vec2 vec);
cv::Point toCvPoint(Vec2i vec);
cv::Point toCvPoint(const Vec2 &vec, double scaleX, double scaleY,
                    cv::Point shift);

template <typename T>
Eigen::Matrix<T, 4, 1> makeHomogeneous(const Eigen::Matrix<T, 3, 1> &v) {
  optimize::Vec4t vH;
  vH.head<3>() = v;
  vH[3] = T(1);
  return vH;
}

cv::Vec3b toCvVec3bDummy(cv::Scalar scalar);

template <typename TT> struct accum_type { using type = TT; };
template <> struct accum_type<unsigned char> { using type = int; };
template <> struct accum_type<signed char> { using type = int; };
template <> struct accum_type<char> { using type = int; };
template <> struct accum_type<cv::Vec3b> { using type = cv::Vec3i; };

template <typename T> cv::Mat boxFilterPyrDown(const cv::Mat &img) {
  constexpr int d = 2;
  cv::Mat result(img.rows / d, img.cols / d, img.type());
  for (int y = 0; y < img.rows / d * d; y += d)
    for (int x = 0; x < img.cols / d * d; x += d) {
      typename accum_type<T>::type accum = typename accum_type<T>::type();
      for (int yy = 0; yy < d; ++yy)
        for (int xx = 0; xx < d; ++xx)
          accum += img.at<T>(y + yy, x + xx);
      result.at<T>(y / d, x / d) = T(accum / (d * d));
    }

  return result;
}

extern template cv::Mat boxFilterPyrDown<unsigned char>(const cv::Mat &img);
extern template cv::Mat boxFilterPyrDown<cv::Vec3b>(const cv::Mat &img);

cv::Mat1b cvtBgrToGray(const cv::Mat &coloredImg);
cv::Mat3b cvtBgrToGray3(const cv::Mat3b coloredImg);
cv::Mat3b cvtGrayToBgr(const cv::Mat1b &grayImg);

cv::Mat1d pyrNUpDepth(const cv::Mat1d &integralWeightedDepths,
                      const cv::Mat1d &integralWeights, int levelNum);

cv::Mat3b drawDepthedFrame(const cv::Mat1b &frame, const cv::Mat1d &depths,
                           double minDepth, double maxDepth);

std::vector<double> readBin(const fs::path &filename);

std::string curTimeBrief();

namespace optimize {

template <int nret, int Options1, int Options2, int Options3, int Options4>
T tangentErr(const Eigen::Matrix<T, nret, 4, Options1> &expected,
             const Eigen::Matrix<T, nret, 4, Options2> &actual,
             const SO3t &diffRotation,
             Eigen::Matrix<T, nret, 3, Options3> &expectedTang,
             Eigen::Matrix<T, nret, 3, Options4> &actualTang) {
  Mat43t dParam = diffRotation.Dx_this_mul_exp_x_at_0();
  expectedTang = expected * dParam;
  actualTang = actual * dParam;
  return (expectedTang - actualTang).norm();
}

} // namespace optimize

} // namespace mdso

#endif
