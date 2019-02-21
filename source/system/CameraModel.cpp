#include "system/CameraModel.h"
#include "util/defs.h"
#include "util/settings.h"
#include "util/types.h"
#include <algorithm>
#include <ceres/ceres.h>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>
#include <vector>

namespace fishdso {

CameraModel::CameraModel(int width, int height, double scale,
                         const Vec2 &center, VecX unmapPolyCoeffs)
    : width(width), height(height), unmapPolyDeg(unmapPolyCoeffs.rows()),
      unmapPolyCoeffs(unmapPolyCoeffs), center(center), scale(scale) {
  normalize();
  setMapPolyCoeffs();
}

CameraModel::CameraModel(int width, int height,
                         const std::string &calibFileName)
    : width(width), height(height) {
  std::ifstream ifs(calibFileName, std::ifstream::in);
  if (!ifs.is_open()) {
    throw std::runtime_error("camera model file could not be open!");
  }
  ifs >> *this;
  normalize();
  setMapPolyCoeffs();
}

CameraModel::CameraModel(int width, int height, double f, double cx, double cy)
    : width(width), height(height), unmapPolyDeg(0), center(cx, cy), scale(1) {
  unmapPolyCoeffs.resize(1, 1);
  unmapPolyCoeffs[0] = f;
  normalize();
  setMapPolyCoeffs();

  std::cout << "unmap coeffs  = " << unmapPolyCoeffs.transpose()
            << "\nmap poly coeffs = " << mapPolyCoeffs.transpose() << std::endl;
}

void CameraModel::normalize() {
  double wd = width, hd = height;
  Vec2 imcenter = center * scale;
  double cornersDist[] = {
      (Vec2(0, 0) - imcenter).norm(), (Vec2(wd, 0) - imcenter).norm(),
      (Vec2(0, hd) - imcenter).norm(), (Vec2(wd, hd) - imcenter).norm()};
  double newScale = *std::max_element(cornersDist, cornersDist + 4);
  double s = newScale / scale;
  double sK = scale / newScale;
  unmapPolyCoeffs[0] *= sK;
  sK = s;
  for (int i = 1; i < unmapPolyDeg; ++i) {
    unmapPolyCoeffs[i] *= sK;
    sK *= s;
  }
  center /= s;
  scale = newScale;
  maxRadius = 1;
  double minZUnnorm = calcUnmapPoly(maxRadius);
  minZ = minZUnnorm / std::hypot(minZUnnorm, maxRadius);
  maxAngle = std::atan2(maxRadius, minZUnnorm);
}

std::pair<Vec2, Mat23> CameraModel::diffMap(const Vec3 &ray) const {
  ceres::Jet<double, 3> rayJet[3];
  for (int i = 0; i < 3; ++i) {
    rayJet[i].a = ray[i];
    rayJet[i].v.setZero();
    rayJet[i].v[i] = 1;
  }
  Eigen::Matrix<ceres::Jet<double, 3>, 2, 1> pointJet = map(rayJet);
  Mat23 mapJacobian;
  mapJacobian << pointJet[0].v.transpose(), pointJet[1].v.transpose();
  return {Vec2(pointJet[0].a, pointJet[1].a), mapJacobian};
}

bool CameraModel::isOnImage(const Vec2 &p, int border) const {
  return p[0] >= border && p[0] < (width - border) && p[1] >= border &&
         p[1] <= (height - border);
}

double CameraModel::getImgRadiusByAngle(double observeAngle) const {
  return scale * calcMapPoly(observeAngle);
}

void CameraModel::getRectByAngle(double observeAngle, int &retWidth,
                                 int &retHeight) const {
  double r = calcMapPoly(observeAngle);
  retWidth = int(r * width / maxRadius);
  retHeight = int(r * height / maxRadius);
}

std::istream &operator>>(std::istream &is, CameraModel &cc) {
  std::string tag;
  is >> tag;
  if (tag != "omnidirectional")
    throw std::runtime_error("Invalid camera type");

  is >> cc.scale >> cc.center[0] >> cc.center[1] >> cc.unmapPolyDeg;
  cc.unmapPolyCoeffs.resize(cc.unmapPolyDeg, 1);

  for (int i = 0; i < cc.unmapPolyDeg; ++i)
    is >> cc.unmapPolyCoeffs[i];

  cc.maxRadius = 2;
  double minZUnnorm = cc.calcUnmapPoly(cc.maxRadius);
  cc.minZ = minZUnnorm / std::hypot(minZUnnorm, cc.maxRadius);
  return is;
}

void CameraModel::setMapPolyCoeffs() {
  // points of type (r, \theta), where r stands for scaled radius of a point
  // in the image and \theta stands for angle to z-axis of the unprojected ray
  int nPnts = settingCameraMapPolyPoints;
  int deg = settingCameraMapPolyDegree;
  StdVector<Vec2> funcGraph;
  funcGraph.reserve(nPnts);
  std::mt19937 gen;
  std::uniform_real_distribution<> distr(0, maxRadius);

  for (int it = 0; it < nPnts; ++it) {
    double r = distr(gen);
    // double r = maxRadius;
    double z = calcUnmapPoly(r);
    double angle = std::atan2(r, z);
    funcGraph.push_back(Vec2(r, angle));
  }

  std::sort(funcGraph.begin(), funcGraph.end(),
            [](Vec2 a, Vec2 b) { return a[0] < b[0]; });

  for (int i = 0; i < int(funcGraph.size()) - 1; ++i)
    if (funcGraph[i][1] > funcGraph[i + 1][1])
      throw std::runtime_error(
          "the camera unmapping polynomial (r -> angle) is not increasing!");

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

  mapPolyCoeffs = A.fullPivHouseholderQr().solve(b);

  //  std::ofstream fx("x" + std::to_string(deg) + ".bin");
  //  fx.write((const char *)mapPolyCoeffs.data(),
  //           mapPolyCoeffs.rows() * mapPolyCoeffs.cols() * sizeof(double));
}

StdVector<CameraModel> CameraModel::camPyr() {
  StdVector<CameraModel> result(settingPyrLevels, *this);
  for (int i = 0; i < settingPyrLevels; ++i) {
    result[i].scale /= (1 << i);
    result[i].width /= (1 << i);
    result[i].height /= (1 << i);
  }

  return result;
}

} // namespace fishdso
