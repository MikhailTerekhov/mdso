#ifndef INCLUDE_CAMERAMODEL
#define INCLUDE_CAMERAMODEL

#include "util/settings.h"
#include "util/types.h"
#include "util/util.h"
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <ceres/ceres.h>
#include <opencv2/core.hpp>
#include <string>

namespace fishdso {

class CameraModel {

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  CameraModel(int width, int height, double scale, const Vec2 &center,
              VecX unmapPolyCoeffs);
  CameraModel(int width, int height, const std::string &calibFileName);
  CameraModel(int width, int height, double f, double cx, double cy);

  template <typename T> Eigen::Matrix<T, 3, 1> unmap(const T *point) const {
    typedef Eigen::Matrix<T, 3, 1> Vec3t;
    typedef Eigen::Matrix<T, 2, 1> Vec2t;
    typedef Eigen::Matrix<T, Eigen::Dynamic, 1> VecXt;
    Eigen::Map<const Vec2t> pt_(point);
    Vec2t pt = pt_;

    VecXt p = unmapPolyCoeffs.cast<T>();
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
    // double expectedR = std::hypot(pt[0], pt[1]);
    // double angle = std::atan2(expectedR, z);
    // double rRecalc = calcMapPoly(angle);
    //    std::cout << "angle = " << angle << std::endl
    //              << "expected r = " << expectedR << std::endl
    //              << "r from zNorm = " << rRecalc << std::endl;
    return res;
  }

  template <typename T> Eigen::Matrix<T, 2, 1> map(const T *point) const {
    typedef Eigen::Matrix<T, 3, 1> Vec3t;
    typedef Eigen::Matrix<T, 2, 1> Vec2t;
    typedef Eigen::Matrix<T, Eigen::Dynamic, 1> VecXt;

    Eigen::Map<const Vec3t> pt_(point);
    Vec3t pt = pt_;
    VecXt p = mapPolyCoeffs.cast<T>();

    T angle = ceres::atan2(pt.template head<2>().norm(), pt[2]);
    T r = p[0];
    T angleN = angle;
    for (int i = 1; i < p.rows(); ++i) {
      r += p[i] * angleN;
      angleN *= angle;
    }

    Vec2t c = center.cast<T>();
    Vec2t res = pt.template head<2>().normalized() * r;
    res += c;
    res *= T(scale);
    return res;
  }

  EIGEN_STRONG_INLINE Vec3 unmap(const Vec2 &point) const {
    return unmap(point.data());
  }

  EIGEN_STRONG_INLINE Vec2 map(const Vec3 &ray) const {
    return map(ray.data());
  }

  template <typename T>
  cv::Mat undistort(const cv::Mat &img, const Mat33 &cameraMatrix) const {
    Mat33 Kinv = cameraMatrix.inverse();
    cv::Mat result = cv::Mat::zeros(img.rows, img.cols, img.type());

    for (int y = 0; y < result.rows; ++y)
      for (int x = 0; x < result.cols; ++x) {
        Vec3 pnt(double(x), double(y), 1.);
        Vec2 origPix = map(Kinv * pnt);
        int origX = origPix[0], origY = origPix[1];
        if (origX >= 0 && origX < result.cols && origY >= 0 &&
            origY < result.rows)
          result.at<T>(y, x) = img.at<T>(origY, origX);
      }
    return result;
  }

  EIGEN_STRONG_INLINE int getWidth() const { return width; }
  EIGEN_STRONG_INLINE int getHeight() const { return height; }
  EIGEN_STRONG_INLINE Vec2 getImgCenter() const { return scale * center; }
  EIGEN_STRONG_INLINE double getMaxAngle() const { return maxAngle; }

  bool isOnImage(const Vec2 &p, int border);

  double getImgRadiusByAngle(double observeAngle) const;
  void getRectByAngle(double observeAngle, int &width, int &height) const;

  void setMapPolyCoeffs();

  StdVector<CameraModel> camPyr();

private:
  friend std::istream &operator>>(std::istream &is, CameraModel &cc);

  EIGEN_STRONG_INLINE double calcUnmapPoly(double r) const {
    double rN = r * r;
    double res = unmapPolyCoeffs[0];
    for (int i = 1; i < unmapPolyDeg; ++i) {
      res += unmapPolyCoeffs[i] * rN;
      rN *= r;
    }
    return res;
  }
  EIGEN_STRONG_INLINE double calcMapPoly(double funcVal) const {
    double funcValN = funcVal;
    double res = mapPolyCoeffs[0];
    for (int i = 1; i < mapPolyCoeffs.rows(); ++i) {
      res += mapPolyCoeffs[i] * funcValN;
      funcValN *= funcVal;
    }
    return res;
  }

  void normalize();

  int width, height;
  int unmapPolyDeg;
  VecX unmapPolyCoeffs;
  Vec2 center;
  double scale;
  double maxRadius;
  double minZ;
  double maxAngle;

  VecX mapPolyCoeffs;
};

} // namespace fishdso

#endif
