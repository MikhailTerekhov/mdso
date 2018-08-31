#pragma once

#include "util/settings.h"
#include "util/types.h"
#include <opencv2/opencv.hpp>
#include <vector>

namespace fishdso {

// global image for debugging purposes
extern cv::Mat dbg;

void putDot(cv::Mat &img, cv::Point const &pos, cv::Scalar const &col);

void grad(cv::Mat const &img, cv::Mat &gradX, cv::Mat &gradY,
          cv::Mat &gradNorm);

cv::Scalar depthCol(double d, double mind, double maxd);

void insertDepths(cv::Mat &img, const std::vector<Vec2> &points,
                  const std::vector<double> &depths, double minDepth,
                  double maxDepth, bool areSolidPnts);

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
        typename accum_type<T>::type val;

        int nonBlackCnt = 0;
        for (int yy = std::max(0, y - d); yy < std::min(y + d, img.rows - 1);
             ++yy)
          for (int xx = std::max(0, x - d); xx < std::min(x + d, img.cols - 1);
               ++xx)
            if (img.at<T>(yy, xx) != T()) {
              val += img.at<T>(yy, xx);
              nonBlackCnt++;
            }
        if (nonBlackCnt != 0)
          img.at<T>(y, x) = T(val / nonBlackCnt);
      }
}

Vec2 toVec2(cv::Point p);
cv::Point toCvPoint(Vec2 vec);
cv::Point toCvPoint(const Vec2 &vec, double scaleX, double scaleY,
                    cv::Point shift);

cv::Vec3b toCvVec3bDummy(cv::Scalar scalar);
} // namespace fishdso
