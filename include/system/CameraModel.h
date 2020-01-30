#ifndef INCLUDE_CAMERAMODEL
#define INCLUDE_CAMERAMODEL

#include "util/settings.h"
#include "util/types.h"
#include "util/util.h"
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <opencv2/core.hpp>
#include <string>

namespace mdso {

class CameraModel {
public:
  using CamPyr = StdVector<CameraModel>;

  enum InputType { POLY_UNMAP, POLY_MAP };

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  CameraModel(int width, int height, double scale, const Vec2 &center,
              VecX unmapPolyCoeffs, const Settings::CameraModel &settings = {});

  CameraModel(int width, int height, const std::string &calibFileName,
              InputType type, const Settings::CameraModel &settings = {});
  CameraModel(int width, int height, double f, double cx, double cy,
              const Settings::CameraModel &settings = {});

  template <typename T> Eigen::Matrix<T, 3, 1> unmap(const T *point) const {
    using Vec3t = Eigen::Matrix<T, 3, 1>;
    using Vec2t = Eigen::Matrix<T, 2, 1>;
    using VecXt = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    Eigen::Map<const Vec2t> pt_(point);
    Vec2t pt = pt_;

    pt[1] = (pt[1] - principalPoint[1]) / fy;
    pt[0] = (pt[0] - skew * pt[1] - principalPoint[0]) / fx;

    VecXt p = unmapPolyCoeffs.cast<T>();

    T rho2 = pt.squaredNorm();
    T rho1 = sqrt(rho2);

    T z = p[0];
    T rhoN = rho2;
    for (int i = 1; i < unmapPolyDeg; i += 2) {
      z += rhoN * p[i];
      if (i + 1 < unmapPolyDeg)
        z += rhoN * rho1 * p[i + 1];
      rhoN *= rho2;
    }

    Vec3t res(pt[0], pt[1], z);
    return res;
  }

  template <typename T> Eigen::Matrix<T, 2, 1> map(const T *point) const {
    typedef Eigen::Matrix<T, 3, 1> Vec3t;
    typedef Eigen::Matrix<T, 2, 1> Vec2t;
    typedef Eigen::Matrix<T, Eigen::Dynamic, 1> VecXt;

    Eigen::Map<const Vec3t> pt_(point);
    Vec3t pt = pt_;
    VecXt p = mapPolyCoeffs.cast<T>();

    T angle = atan2(pt.template head<2>().norm(), pt[2]);

    T r = p[0];
    T angleN = angle;
    for (int i = 1; i < p.rows(); ++i) {
      r += p[i] * angleN;
      angleN *= angle;
    }

    Vec2t res = pt.template head<2>().normalized() * r;
    res[0] = T(fx) * res[0] + T(skew) * res[1] + T(principalPoint[0]);
    res[1] = T(fy) * res[1] + T(principalPoint[1]);
    return res;
  }

  template <typename T>
  inline Eigen::Matrix<T, 3, 1>
  unmap(const Eigen::Matrix<T, 2, 1> &point) const {
    return unmap(point.data());
  }

  template <typename T>
  inline Eigen::Matrix<T, 2, 1> map(const Eigen::Matrix<T, 3, 1> &ray) const {
    return map(ray.data());
  }

  template <typename T>
  bool isMappable(const Eigen::Matrix<T, 3, 1> &ray) const {
    T angle = atan2(ray.template head<2>().norm(), ray[2]);
    return angle < maxAngle;
  }

  template <typename T>
  cv::Mat undistort(const cv::Mat &img, const Mat33 &cameraMatrix) const {
    Mat33 Kinv = cameraMatrix.inverse();
    cv::Mat result = cv::Mat::zeros(img.rows, img.cols, img.type());

    for (int y = 0; y < result.rows; ++y)
      for (int x = 0; x < result.cols; ++x) {
        Vec3 pnt(double(x), double(y), 1.);
        Vec2 origPix = map((Kinv * pnt).eval());
        int origX = origPix[0], origY = origPix[1];
        if (origX >= 0 && origX < result.cols && origY >= 0 &&
            origY < result.rows)
          result.at<T>(y, x) = img.at<T>(origY, origX);
      }
    return result;
  }

  std::pair<Vec2, Mat23> diffMap(const Vec3 &ray) const;
  std::pair<Vec2f, Mat23f> diffMap(const Vec3f &ray) const;

  inline int getWidth() const { return width; }
  inline int getHeight() const { return height; }
  inline Vec2 getImgCenter() const { return principalPoint; }
  inline double getMinZ() const { return minZ; }
  inline double getMaxAngle() const { return maxAngle; }

  void setMask(const cv::Mat1b &mask);
  const cv::Mat1b &mask() const { return mMask; };

  bool isOnImage(const Vec2 &p, int border) const;

  double getImgRadiusByAngle(double observeAngle) const;
  void getRectByAngle(double observeAngle, int &width, int &height) const;

  void setImageCenter(const Vec2 &imcenter);

  void setMapPolyCoeffs();
  void setUnmapPolyCoeffs();

  CamPyr camPyr(int pyrLevels);

private:
  void readUnmap(std::istream &is);
  void readMap(std::istream &is);

  inline double calcUnmapPoly(double r) const {
    double rN = r * r;
    double res = unmapPolyCoeffs[0];
    for (int i = 1; i < unmapPolyDeg; ++i) {
      res += unmapPolyCoeffs[i] * rN;
      rN *= r;
    }
    return res;
  }
  inline double calcMapPoly(double funcVal) const {
    double funcValN = funcVal;
    double res = mapPolyCoeffs[0];
    for (int i = 1; i < mapPolyCoeffs.rows(); ++i) {
      res += mapPolyCoeffs[i] * funcValN;
      funcValN *= funcVal;
    }
    return res;
  }

  // void normalize();

  void recalcMaxRadius();
  void recalcBoundaries();

  int width, height;
  int unmapPolyDeg;
  VecX unmapPolyCoeffs;
  double fx, fy;
  Vec2 principalPoint;
  double skew;
  double maxRadius;
  double minZ;
  double maxAngle;
  VecX mapPolyCoeffs;
  Settings::CameraModel settings;
  cv::Mat1b mMask;
};

} // namespace mdso

#endif
