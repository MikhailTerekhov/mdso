#ifndef INCLUDE_UTIL
#define INCLUDE_UTIL

#include "util/settings.h"
#include "util/types.h"
#include <opencv2/opencv.hpp>
#include <vector>

namespace fishdso {

// global image for debugging purposes
extern cv::Mat dbg;
extern double minDepth, maxDepth;

void putDot(cv::Mat &img, const cv::Point &pos, const cv::Scalar &col);
void putCross(cv::Mat &img, const cv::Point &pos, const cv::Scalar &col,
              int size, int thikness);

void grad(cv::Mat const &img, cv::Mat &gradX, cv::Mat &gradY,
          cv::Mat &gradNorm);

cv::Scalar depthCol(double d, double mind, double maxd);

void insertDepths(cv::Mat &img, const StdVector<Vec2> &points,
                  const std::vector<double> &depths, double minDepth,
                  double maxDepth, bool areSolidPnts);

Vec2 toVec2(cv::Point p);
cv::Point toCvPoint(Vec2 vec);
cv::Point toCvPoint(const Vec2 &vec, double scaleX, double scaleY,
                    cv::Point shift);

cv::Vec3b toCvVec3bDummy(cv::Scalar scalar);

template <typename TT> struct accum_type { typedef TT type; };
template <> struct accum_type<unsigned char> { typedef int type; };
template <> struct accum_type<signed char> { typedef int type; };
template <> struct accum_type<char> { typedef int type; };
template <> struct accum_type<cv::Vec3b> { typedef cv::Vec3i type; };

template <typename T> void fillBlackPixels(cv::Mat &img) {
  int d = settingHalfFillingFilterSize;
  for (int y = 0; y < img.rows; ++y)
    for (int x = 0; x < img.cols; ++x)
      if (img.at<T>(y, x) == T()) {
        typename accum_type<T>::type accum = typename accum_type<T>::type();

        int nonBlackCnt = 0;
        for (int yy = std::max(0, y - d); yy < std::min(y + d, img.rows - 1);
             ++yy)
          for (int xx = std::max(0, x - d); xx < std::min(x + d, img.cols - 1);
               ++xx)
            if (img.at<T>(yy, xx) != T()) {
              accum += img.at<T>(yy, xx);
              nonBlackCnt++;
            }
        if (nonBlackCnt != 0)
          img.at<T>(y, x) = T(accum / nonBlackCnt);
      }
}

template <typename T> cv::Mat boxFilterPyrUp(const cv::Mat &img) {
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

extern template cv::Mat boxFilterPyrUp<unsigned char>(const cv::Mat &img);
extern template cv::Mat boxFilterPyrUp<cv::Vec3b>(const cv::Mat &img);

cv::Mat pyrNUpDepth(const cv::Mat1d &integralWeightedDepths,
                    const cv::Mat1d &integralWeights, int levelNum);

cv::Mat drawDepthedFrame(const cv::Mat1b &frame, const cv::Mat1d &depths,
                         double minDepth, double maxDepth);

} // namespace fishdso

#endif
