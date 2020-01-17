#include "system/CameraModel.h"
#include "util/defs.h"
#include "util/flags.h"
#include "util/settings.h"
#include "util/types.h"
#include <algorithm>
#include <ceres/ceres.h>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>
#include <vector>

namespace mdso {

CameraModel::CameraModel(int width, int height, double scale,
                         const Vec2 &center, VecX unmapPolyCoeffs,
                         const Settings::CameraModel &settings)
    : width(width)
    , height(height)
    , unmapPolyDeg(unmapPolyCoeffs.rows())
    , unmapPolyCoeffs(unmapPolyCoeffs)
    , fx(scale)
    , fy(scale)
    , principalPoint(fx * center[0], fy * center[1])
    , skew(0)
    , settings(settings)
    , mMask(height, width, CV_WHITE_BYTE) {
  recalcBoundaries();
  setMapPolyCoeffs();
}

CameraModel::CameraModel(int width, int height,
                         const std::string &calibFileName, InputType type,
                         const Settings::CameraModel &settings)
    : width(width)
    , height(height)
    , skew(0)
    , settings(settings)
    , mMask(height, width, CV_WHITE_BYTE) {
  std::ifstream ifs(calibFileName, std::ifstream::in);
  if (!ifs.is_open())
    throw std::runtime_error("camera model file could not be open!");

  switch (type) {
  case POLY_UNMAP:
    readUnmap(ifs);
    recalcBoundaries();
    setMapPolyCoeffs();
    break;
  case POLY_MAP:
    readMap(ifs);
    maxAngle = settings.magicMaxAngle;
    setUnmapPolyCoeffs();
    recalcBoundaries();
    break;
  }
}

CameraModel::CameraModel(int width, int height, double f, double cx, double cy,
                         const Settings::CameraModel &settings)
    : width(width)
    , height(height)
    , unmapPolyDeg(0)
    , fx(f)
    , fy(f)
    , principalPoint(f * cx, f * cy)
    , settings(settings)
    , mMask(height, width, CV_WHITE_BYTE) {
  unmapPolyCoeffs.resize(1, 1);
  unmapPolyCoeffs[0] = f;
  recalcBoundaries();

  setMapPolyCoeffs();

  LOG(INFO) << "\n\n CAMERA MODEL:\n";
  LOG(INFO) << "unmap coeffs  = " << unmapPolyCoeffs.transpose() << "\n";
  LOG(INFO) << "\nmap poly coeffs = " << mapPolyCoeffs.transpose() << "\n\n";
}

bool CameraModel::isMappable(const Vec3 &point) const {
  double angle = atan2(point.head<2>().norm(), point[2]);
  return angle < maxAngle;
}

template <typename T>
std::pair<Eigen::Matrix<T, 2, 1>, Eigen::Matrix<T, 2, 3>>
diffMapHelper(const CameraModel *cam, const Eigen::Matrix<T, 3, 1> &ray) {
  using Vec2t = Eigen::Matrix<T, 2, 1>;
  using Mat23t = Eigen::Matrix<T, 2, 3>;
  ceres::Jet<T, 3> rayJet[3];
  for (int i = 0; i < 3; ++i) {
    rayJet[i].a = ray[i];
    rayJet[i].v.setZero();
    rayJet[i].v[i] = 1;
  }
  Eigen::Matrix<ceres::Jet<T, 3>, 2, 1> pointJet = cam->map(rayJet);
  Mat23t mapJacobian;
  mapJacobian << pointJet[0].v.transpose(), pointJet[1].v.transpose();
  return {Vec2t(pointJet[0].a, pointJet[1].a), mapJacobian};
}

std::pair<Vec2, Mat23> CameraModel::diffMap(const Vec3 &ray) const {
  return diffMapHelper(this, ray);
}

std::pair<Vec2f, Mat23f> CameraModel::diffMap(const Vec3f &ray) const {
  return diffMapHelper(this, ray);
}

bool CameraModel::isOnImage(const Vec2 &p, int border) const {
  bool inBorder = Eigen::AlignedBox2d(Vec2(border, border),
                                      Vec2(width - border, height - border))
                      .contains(p);
  return inBorder && mMask(toCvPoint(p));
}

double CameraModel::getImgRadiusByAngle(double observeAngle) const {
  double mapped = calcMapPoly(observeAngle);
  return std::min(fx * mapped, fy * mapped);
}

void CameraModel::getRectByAngle(double observeAngle, int &retWidth,
                                 int &retHeight) const {
  double r = calcMapPoly(observeAngle);
  retWidth = int(r * width / maxRadius);
  retHeight = int(r * height / maxRadius);
}

void CameraModel::setImageCenter(const Vec2 &imcenter) {
  principalPoint = imcenter;
}

void CameraModel::readUnmap(std::istream &is) {
  std::string tag;
  is >> tag;
  if (tag != "omnidirectional")
    throw std::runtime_error("Invalid camera type");

  double scale, cx, cy;

  is >> scale >> cx >> cy >> unmapPolyDeg;
  fx = fy = scale;
  principalPoint = Vec2(fx * cx, fy * cy);
  unmapPolyCoeffs.resize(unmapPolyDeg, 1);

  for (int i = 0; i < unmapPolyDeg; ++i)
    is >> unmapPolyCoeffs[i];
}

void CameraModel::readMap(std::istream &is) {
  int mapPolyDeg = 0;
  is >> mapPolyDeg >> fx >> fy >> principalPoint[0] >> principalPoint[1] >>
      skew;
  // CHECK(std::abs(skew) < 0.05) << "Our CameraModel expects zero skew";
  mapPolyCoeffs.resize(mapPolyDeg + 2);
  mapPolyCoeffs[0] = 0;
  mapPolyCoeffs[1] = 1;
  for (int i = 0; i < mapPolyDeg; ++i)
    is >> mapPolyCoeffs[i + 2];
}

void CameraModel::setMapPolyCoeffs() {
  // points of type (r, \theta), where r stands for scaled radius of a point
  // in the image and \theta stands for angle to z-axis of the unprojected ray
  int nPnts = settings.mapPolyPoints;
  int deg = settings.mapPolyDegree;
  StdVector<Vec2> funcGraph;
  funcGraph.reserve(nPnts);
  std::mt19937 gen(FLAGS_deterministic ? 42 : std::random_device()());
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

  mapPolyCoeffs = A.fullPivHouseholderQr().solve(b);
}

void CameraModel::setUnmapPolyCoeffs() {
  unmapPolyDeg = settings.unmapPolyDegree;

  StdVector<std::pair<double, double>> funcGraph;
  funcGraph.reserve(settings.unmapPolyPoints);

  std::mt19937 gen;
  const double eps = 1e-3;
  std::uniform_real_distribution<double> distr(eps, maxAngle);
  for (int i = 0; i < settings.unmapPolyPoints; ++i) {
    double theta = distr(gen);
    double r = calcMapPoly(theta);
    double z = r * std::tan(M_PI_2 - theta);
    funcGraph.push_back({r, z});
  }

  MatXX A(funcGraph.size(), settings.unmapPolyDegree);
  VecX b(funcGraph.size());
  for (int i = 0; i < funcGraph.size(); ++i) {
    A(i, 0) = 1;
    const double &r = funcGraph[i].first;
    double rN = r * r;
    for (int j = 1; j < settings.unmapPolyDegree; ++j) {
      A(i, j) = rN;
      rN *= r;
    }
    b(i) = funcGraph[i].second;
  }

  unmapPolyCoeffs = A.fullPivHouseholderQr().solve(b);
  std::cout << "coeffs: " << unmapPolyCoeffs.transpose() << '\n';
}

cv::Mat1b downsampleMask(const cv::Mat1b &mask, int lvl) {
  int scale = (1 << lvl);
  cv::Mat1b result(mask.rows / scale, mask.cols / scale, 1);
  for (int y = 0; y < mask.rows; ++y)
    for (int x = 0; x < mask.cols; ++x)
      if (mask(y, x) == 0)
        result(y / scale, x / scale) = false;
  return result;
}

CameraModel::CamPyr CameraModel::camPyr(int pyrLevels) {
  CHECK(pyrLevels > 0 && pyrLevels <= Settings::Pyramid::max_levelNum);
  CamPyr result;
  for (int i = 0; i < pyrLevels; ++i) {
    result.emplace_back(*this);
    result.back().width /= (1 << i);
    result.back().height /= (1 << i);
    result.back().principalPoint /= (1 << i);
    result.back().fx /= (1 << i);
    result.back().fy /= (1 << i);
    result.back().setMask(downsampleMask(mMask, i));
  }

  return result;
}

void CameraModel::recalcMaxRadius() {
  Vec2 normPp(principalPoint[0] / fx, principalPoint[1] / fy);
  Vec2 normSz(width / fx, height / fy);
  double dist[] = {normPp.norm(), (normPp - Vec2(0, normSz[1])).norm(),
                   (normPp - Vec2(normSz[0], 0)).norm(),
                   (normPp - normSz).norm()};
  maxRadius = *std::max_element(dist, dist + 4);
}

void CameraModel::recalcBoundaries() {
  recalcMaxRadius();
  double minZUnnorm = calcUnmapPoly(maxRadius);
  minZ = minZUnnorm / std::hypot(minZUnnorm, maxRadius);
  maxAngle = std::atan2(maxRadius, minZUnnorm);
}

void CameraModel::setMask(const cv::Mat1b &mask) { this->mMask = mask; }

} // namespace mdso
