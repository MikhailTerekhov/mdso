#include "cameramodel.h"
#include "../util/defs.h"
#include "../util/settings.h"
#include "../util/types.h"
#include <algorithm>
#include <ceres/ceres.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>
#include <vector>

namespace fishdso {

CameraModel::CameraModel()
    : unmapPolyDeg(0), center(Vec2(0.0, 0.0)), scale(1.0), maxRadius(1.0),
      minZ(-1), maxAngle(M_PI), isNormalized(false) {}

CameraModel::CameraModel(int unmapPolyDeg, const VecX &unmapPolyCoefs,
                         double scale, double maxRadius, const Vec2 &center)
    : unmapPolyDeg(unmapPolyDeg), unmapPolyCoefs(unmapPolyCoefs),
      center(center), scale(scale), maxRadius(maxRadius),
      minZ(calcUnmapPoly(maxRadius)), maxAngle(std::atan2(maxRadius, minZ)),
      isNormalized(false) {
  setMapPolyCoefs();
}

void CameraModel::normalize(int imgWidth, int imgHeight) {
  double newScale = std::hypot(imgWidth, imgHeight) / 2;
  double s = newScale / scale;
  double sK = scale / newScale;
  unmapPolyCoefs[0] *= sK;
  sK = s;
  for (int i = 1; i < unmapPolyDeg; ++i) {
    unmapPolyCoefs[i] *= sK;
    sK *= s;
  }
  center /= s;
  scale = newScale;
  maxRadius = 1;
  double minZUnnorm = calcUnmapPoly(maxRadius);
  minZ = minZUnnorm / std::hypot(minZUnnorm, maxRadius);
  maxAngle = std::atan2(maxRadius, minZUnnorm);
  isNormalized = true;
  setMapPolyCoefs();
}

void CameraModel::undistort(const cv::Mat &img, cv::Mat &result,
                            const Mat33 &cameraMatrix) {
  img.copyTo(result);
  result.setTo(CV_BLACK);
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
        if (result.channels() == 3)
          result.at<cv::Vec3b>(newY, newX) = img.at<cv::Vec3b>(y, x);
        else if (result.channels() == 1)
          result.at<unsigned char>(newY, newX) = img.at<unsigned char>(y, x);
    }
}

std::istream &operator>>(std::istream &is, CameraModel &cc) {
  std::string tag;
  is >> tag;
  if (tag != "omnidirectional")
    throw std::runtime_error("Invalid camera type");

  is >> cc.scale >> cc.center[0] >> cc.center[1] >> cc.unmapPolyDeg;
  cc.unmapPolyCoefs.resize(cc.unmapPolyDeg, 1);

  for (int i = 0; i < cc.unmapPolyDeg; ++i)
    is >> cc.unmapPolyCoefs[i];

  cc.maxRadius = 2;
  double minZUnnorm = cc.calcUnmapPoly(cc.maxRadius);
  cc.minZ = minZUnnorm / std::hypot(minZUnnorm, cc.maxRadius);
  cc.setMapPolyCoefs();
  return is;
}

EIGEN_STRONG_INLINE double CameraModel::calcUnmapPoly(double r) {
  double rN = r * r;
  double res = unmapPolyCoefs[0];
  for (int i = 1; i < unmapPolyDeg; ++i) {
    res += unmapPolyCoefs[i] * rN;
    rN *= r;
  }
  return res;
}

EIGEN_STRONG_INLINE double CameraModel::calcMapPoly(double funcVal) {
  double funcValN = funcVal;
  double res = mapPolyCoefs[0];
  for (int i = 1; i < mapPolyCoefs.rows(); ++i) {
    res += mapPolyCoefs[i] * funcValN;
    funcValN *= funcVal;
  }
  return res;
}

void CameraModel::setMapPolyCoefs() {
  // points of type (\tilde{z}, r), where r stands for scaled radius of a point
  // in the image and \tilde{z} stands for z-coordinate of the normalized
  // unprojected ray
  int nPnts = settingCameraMapPolyPoints;
  int deg = settingCameraMapPolyDegree;
  std::vector<Vec2> funcGraph;
  funcGraph.reserve(nPnts);
  std::mt19937 gen;
  std::uniform_real_distribution<> distr(0, maxRadius);

#if CAMERA_MAP_TYPE == CAMERA_MAP_POLYNOMIAL_Z
  for (int it = 0; it < nPnts; ++it) {
    double r = distr(gen);
    // double r = maxRadius;
    double z = calcUnmapPoly(r);

    double abs = std::hypot(r, z);
    if (abs > 1e-3)
      funcGraph.push_back(Vec2(r, z / abs));
    else
      funcGraph.push_back(Vec2(0, 1));
  }

  std::sort(funcGraph.begin(), funcGraph.end(),
            [](Vec2 a, Vec2 b) { return a[0] < b[0]; });

  for (int i = 0; i < funcGraph.size() - 1; ++i)
    if (funcGraph[i][1] < funcGraph[i + 1][1])
      throw std::runtime_error(
          "the camera unmapping polynomial (r -> zNorm) is not decreasing!");
#elif CAMERA_MAP_TYPE == CAMERA_MAP_POLYNOMIAL_ANGLE
  for (int it = 0; it < nPnts; ++it) {
    double r = distr(gen);
    // double r = maxRadius;
    double z = calcUnmapPoly(r);
    double angle = std::atan2(r, z);
    funcGraph.push_back(Vec2(r, angle));
  }

  std::sort(funcGraph.begin(), funcGraph.end(),
            [](Vec2 a, Vec2 b) { return a[0] < b[0]; });

  for (int i = 0; i < funcGraph.size() - 1; ++i)
    if (funcGraph[i][1] > funcGraph[i + 1][1])
      throw std::runtime_error(
          "the camera unmapping polynomial (r -> angle) is not increasing!");
#endif

  // solving linear least-squares to approximate func^(-1) with a polynomial
  MatXX A(nPnts, deg + 1);
  VecX b(nPnts);
  for (int i = 0; i < nPnts; ++i) {
    const double &funcVal = funcGraph[i][1];
    double funcValN = 1;
    for (int j = 0; j <= deg; ++j) {
      A(i, j) = funcValN;
      funcValN *= funcVal;
    }
    b(i) = funcGraph[i][0];
  }

  mapPolyCoefs = A.fullPivHouseholderQr().solve(b);
}

void CameraModel::testMapPoly() {
  const int testnum = 200;
  double sqErr = 0.0;
  std::mt19937 gen;
  std::uniform_real_distribution<> distr(0, maxRadius);

#if CAMERA_MAP_TYPE == CAMERA_MAP_POLYNOMIAL_Z
  for (int i = 0; i < testnum; ++i) {
    double r = distr(gen);
    double zUnnorm = calcUnmapPoly(r);
    double z = zUnnorm / std::hypot(r, zUnnorm);
    double rBack = calcMapPoly(z);
    sqErr += (r - rBack) * (r - rBack);
  }
#elif CAMERA_MAP_TYPE == CAMERA_MAP_POLYNOMIAL_ANGLE
  for (int i = 0; i < testnum; ++i) {
    double r = distr(gen);
    double zUnnorm = calcUnmapPoly(r);
    double angle = std::atan2(r, zUnnorm);
    double rBack = calcMapPoly(angle);
    sqErr += (r - rBack) * (r - rBack);
  }
#endif
  std::cout << "mapPoly rmse = " << std::sqrt(sqErr / testnum) << std::endl;
}

void CameraModel::testReproject() {
  const int testnum = 200;
  int width = 1920, height = 1208;
  double sqErr = 0.0;
  for (int i = 0; i < testnum; ++i) {
    Vec2 pnt(double(rand() % width), double(rand() % height));
    Vec3 ray = unmap(pnt.data());
    ray *= 10.5;

    Vec2 pntBack = map(ray.data());
    sqErr += (pnt - pntBack).squaredNorm();
  }
  std::cout << "reproject rmse (pix) = " << std::sqrt(sqErr / testnum)
            << std::endl;
}

} // namespace fishdso
