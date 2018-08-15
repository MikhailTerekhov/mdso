#include "cameramodel.h"
#include "../util/defs.h"
#include "../util/settings.h"
#include "../util/types.h"
#include <algorithm>
#include <ceres/ceres.h>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>
#include <vector>

namespace fishdso {

CameraModel::CameraModel(int width, int height,
                         const std::string &calibFileName)
    : width(width), height(height) {
  std::ifstream ifs(calibFileName, std::ifstream::in);
  ifs >> *this;
  normalize();
  setMapPolyCoefs();
}

int CameraModel::getWidth() const { return width; }

int CameraModel::getHeight() const { return height; }

void CameraModel::normalize() {
  double newScale = std::hypot(width, height) / 2;
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
}

void CameraModel::getRectByAngle(double observeAngle, int &retWidth,
                                 int &retHeight) const {
#if CAMERA_MAP_TYPE == CAMERA_MAP_POLYNOMIAL_ANGLE
  double r = calcMapPoly(observeAngle);
  retWidth = int(r * width / maxRadius);
  retHeight = int(r * height / maxRadius);
#else
  thow std::runtime_error("getRectByAngle with wrong map type!");
#endif
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
  return is;
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

  //  std::ofstream fA("A" + std::to_string(deg) + ".bin");
  //  fA.write((const char *)A.data(), A.rows() * A.cols() * sizeof(double));
  //  std::ofstream fb("b" + std::to_string(deg) + ".bin");
  //  fb.write((const char *)b.data(), b.rows() * b.cols() * sizeof(double));

  mapPolyCoefs = A.fullPivHouseholderQr().solve(b);

  //  std::ofstream fx("x" + std::to_string(deg) + ".bin");
  //  fx.write((const char *)mapPolyCoefs.data(),
  //           mapPolyCoefs.rows() * mapPolyCoefs.cols() * sizeof(double));
}

void CameraModel::testMapPoly() const {
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
  std::cout << std::sqrt(sqErr / testnum) << " coefs = ";
  for (int i = 0; i < mapPolyCoefs.rows(); ++i)
    std::cout << mapPolyCoefs[i] << ' ';
  std::cout << std::endl;
}

void CameraModel::testReproject() {
  const int testnum = 2000;
  int width = 1920, height = 1208;
  double sqErr = 0.0;

  for (int i = 0; i < testnum; ++i) {
    Vec2 pnt(double(rand() % width), double(rand() % height));
    Vec3 ray = unmap(pnt.data());
    ray *= 10.5;

    Vec2 pntBack = map(ray.data());
    sqErr += (pnt - pntBack).squaredNorm();
  }
  //  std::cout << "reproject rmse (pix) = " << std::sqrt(sqErr / testnum)
  //            << std::endl;
  std::cout << std::sqrt(sqErr / testnum) << "; coefs = ";
  for (int i = 0; i < mapPolyCoefs.rows(); ++i)
    std::cout << mapPolyCoefs[i] << ' ';
  std::cout << std::endl;
}

} // namespace fishdso
