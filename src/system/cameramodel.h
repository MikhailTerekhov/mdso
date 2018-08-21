#pragma once

#include "../util/settings.h"
#include "../util/types.h"
#include "../util/util.h"
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <opencv2/core.hpp>
#include <string>

namespace fishdso {

class CameraModel {

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  CameraModel(int width, int height, double scale, Vec2 center,
              VecX unmapPolyCoefs);
  CameraModel(int width, int height, const std::string &calibFileName);

  template <typename T> Eigen::Matrix<T, 3, 1> unmap(const T *point) const {
    typedef Eigen::Matrix<T, 3, 1> Vec3t;
    typedef Eigen::Matrix<T, 2, 1> Vec2t;
    typedef Eigen::Matrix<T, Eigen::Dynamic, 1> VecXt;
    Eigen::Map<const Vec2t> pt_(point);
    Vec2t pt = pt_;

    VecXt p = unmapPolyCoefs.cast<T>();
    Vec2t c = center.cast<T>();

    pt /= scale;
    pt -= c;

    T rho2 = pt.squaredNorm();
    T rho1 = ceres::sqrt(rho2);

    T z = p[0];
    T rhoN = rho2;
    for (int i = 1; i < unmapPolyDeg; i += 2) {
      z += rhoN * p[i];
      if (i + 1 < unmapPolyDeg)
        z += rhoN * rho1 * p[i + 1];
      rhoN *= rho2;
    }

    Vec3t res(pt[0], pt[1], z);
    double expectedR = std::hypot(pt[0], pt[1]);
    double angle = std::atan2(expectedR, z);
    double rRecalc = calcMapPoly(angle);
    //    std::cout << "angle = " << angle << std::endl
    //              << "expected r = " << expectedR << std::endl
    //              << "r from zNorm = " << rRecalc << std::endl;
    return res;
  }

  template <typename T> Eigen::Matrix<T, 2, 1> map(const T *point) const {
    typedef Eigen::Matrix<T, 3, 1> Vec3t;
    typedef Eigen::Matrix<T, 2, 1> Vec2t;
    typedef Eigen::Matrix<T, Eigen::Dynamic, 1> VecXt;

#if CAMERA_MAP_TYPE == CAMERA_MAP_POLYNOMIAL_Z
    Eigen::Map<const Vec3t> pt_(point);
    Vec3t pt = pt_;
    VecXt p = mapPolyCoefs.cast<T>();
    T z2 = pt[2] * pt[2] / pt.squaredNorm();
    T z1 = ceres::sqrt(z2);
    if (pt[2] < 0)
      z1 = -z1;
    T zN = T(1.0);
    T r = p[0];
    for (int i = 1; i < p.rows(); i += 2) {
      r += p[i] * zN * z1;
      zN *= z2;
      if (i + 1 <= p.rows())
        r += p[i + 1] * zN;
    }

    Vec2t c = center.cast<T>();
    Vec2t res = pt.template head<2>().normalized() * r;
    res += c;
    res *= scale;
    return res;
#elif CAMERA_MAP_TYPE == CAMERA_MAP_POLYNOMIAL_ANGLE
    Eigen::Map<const Vec3t> pt_(point);
    Vec3t pt = pt_;
    VecXt p = mapPolyCoefs.cast<T>();

    T angle = ceres::atan2(pt.template head<2>().norm(), pt[2]);
    T r = mapPolyCoefs[0];
    T angleN = angle;
    for (int i = 1; i < p.rows(); ++i) {
      r += mapPolyCoefs[i] * angleN;
      angleN *= angle;
    }

    Vec2t c = center.cast<T>();
    Vec2t res = pt.template head<2>().normalized() * r;
    res += c;
    res *= scale;
    return res;
#endif
  }

  template <typename T>
  void undistort(const cv::Mat &img, cv::Mat &result,
                 const Mat33 &cameraMatrix) const {
    result = cv::Mat::zeros(img.rows, img.cols, img.type());
    double pnt[] = {0, 0};
    for (int y = 0; y < img.rows; ++y)
      for (int x = 0; x < img.cols; ++x) {
        pnt[0] = x;
        pnt[1] = y;
        Vec3 newPixelD = cameraMatrix * unmap(pnt);
        int newX = int(newPixelD[0] / newPixelD[2]);
        int newY = int(newPixelD[1] / newPixelD[2]);
        if (newPixelD[2] > 0 && newX >= 0 && newX < result.cols && newY >= 0 &&
            newY < result.rows)
          result.at<T>(newY, newX) = img.at<T>(y, x);
      }
    fillBlackPixels<T>(result);
  }

  int getWidth() const;
  int getHeight() const;

  void getRectByAngle(double observeAngle, int &width, int &height) const;

  void setMapPolyCoefs();

  void testMapPoly() const;
  void testReproject();

private:
  friend std::istream &operator>>(std::istream &is, CameraModel &cc);

  EIGEN_STRONG_INLINE double calcUnmapPoly(double r) const {
    double rN = r * r;
    double res = unmapPolyCoefs[0];
    for (int i = 1; i < unmapPolyDeg; ++i) {
      res += unmapPolyCoefs[i] * rN;
      rN *= r;
    }
    return res;
  }
  EIGEN_STRONG_INLINE double calcMapPoly(double funcVal) const {
    double funcValN = funcVal;
    double res = mapPolyCoefs[0];
    for (int i = 1; i < mapPolyCoefs.rows(); ++i) {
      res += mapPolyCoefs[i] * funcValN;
      funcValN *= funcVal;
    }
    return res;
  }

  void normalize();

  int width, height;
  int unmapPolyDeg;
  VecX unmapPolyCoefs;
  Vec2 center;
  double scale;
  double maxRadius;
  double minZ;
  double maxAngle;

  VecX mapPolyCoefs;
};

} // namespace fishdso
