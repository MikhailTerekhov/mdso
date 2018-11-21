#include "system/ImmaturePoint.h"
#include "util/defs.h"
#include "util/util.h"
#include "util/geometry.h"
#include <ceres/internal/autodiff.h>
#include <ceres/jet.h>

namespace fishdso {

ImmaturePoint::ImmaturePoint(PreKeyFrame *baseFrame, const Vec2 &p)
    : p(p), minDepth(0), maxDepth(INF),
      quality(-1), baseFrame(baseFrame), cam(baseFrame->cam) {
  for (int i = 0; i < settingResidualPatternSize; ++i) {
    baseDirections[i] = cam->unmap(p + settingResidualPattern[i]).normalized();
    baseIntencities[i] =
        baseFrame->frame()(toCvPoint(p + settingResidualPattern[i]));
  }
}

// While searching along epipolar curve, we will continously map rays on a
// diametrical segment of a sphere. Since our camera model remains valid only
// when angle between the mapped ray and Oz is smaller then certain maxAngle, we
// want to intersect the segment of search with the "well-mapped" part of the
// sphere, i.e. z > z0.
bool forceCamValidity(double maxObserveAngle, Vec3 &dir1, Vec3 &dir2) {
  double angle1 = angle(dir1, Vec3(0., 0., 1.));
  double angle2 = angle(dir2, Vec3(0., 0., 1.));
  if (angle1 > maxObserveAngle && angle2 > maxObserveAngle)
    return false;
  if (angle1 < maxObserveAngle && angle2 < maxObserveAngle)
    return true;

  double z0 = std::cos(maxObserveAngle);
  double R = std::sin(maxObserveAngle);
  Vec3 norm = dir1.cross(dir2);
  double a = norm[0], b = norm[1], c = norm[2];
  double a2b2 = a * a + b * b;
  double x0 = -(a * c * z0) / a2b2, y0 = -(b * c * z0) / a2b2;
  double u = a * y0 - b * x0, v = x0 * x0 + y0 * y0 - R * R;
  double sD4 = std::sqrt(u * u - a2b2 * v);
  double t1 = (u - sD4) / a2b2, t2 = (u + sD4) / a2b2;
  Vec3 sol1(x0 - t1 * b, y0 + t1 * a, z0), sol2(x0 - t2 * b, y0 + t2 * a, z0);
  double sol1Norm2 = sol1.squaredNorm(), sol2Norm2 = sol2.squaredNorm();
  CHECK(std::abs(sol1Norm2 - 1.0) < 1e-4)
      << "forceCamValidity failed! |sol1| = " << std::sqrt(sol1Norm2)
      << std::endl;
  CHECK(std::abs(sol2Norm2 - 1.0) < 1e-4)
      << "forceCamValidity failed! |sol2| = " << std::sqrt(sol2Norm2)
      << std::endl;
  Mat32 M;
  M << dir1, dir2;
  Vec2 coeffs = M.fullPivHouseholderQr().solve(sol1);
  Vec3 &sol = (coeffs[0] >= 0 && coeffs[1] >= 0) ? sol1 : sol2;

  if (angle1 > maxObserveAngle)
    dir1 = sol;
  else if (angle2 > maxObserveAngle)
    dir2 = sol;

  return true;
}

void ImmaturePoint::traceOn(const PreKeyFrame &refFrame, bool debugOut) {
  AffineLightTransform<double> lightRefToBase =
      baseFrame->lightWorldToThis * refFrame.lightWorldToThis.inverse();
  SE3 baseToRef = refFrame.worldToThis * baseFrame->worldToThis.inverse();
  Vec3 dirMin = (baseToRef * (minDepth * baseDirections[0])).normalized();
  Vec3 dirMax = maxDepth == INF
                    ? baseToRef.so3() * baseDirections[0]
                    : (baseToRef * (maxDepth * baseDirections[0])).normalized();
  if (M_PI - angle(dirMin, dirMax) < 1e-3)
    return;

  if (!forceCamValidity(cam->getMaxAngle(), dirMin, dirMax))
    return;

  cv::Mat base;
  cv::Mat curved;
  if (debugOut) {
    base = baseFrame->frame().clone();
    curved = refFrame.frame().clone();

    putDot(base, toCvPoint(p), CV_BLACK_BYTE);
    cv::imshow("base frame", base);
  }

  bool pointFound;
  double alpha0;
  for (int it = 0; it < settingEpipolarOnImageTestCount; ++it) {
    double alpha = it / (settingEpipolarOnImageTestCount - 1);
    Vec3 curDir = (1 - alpha) * dirMax + alpha * dirMin;
    Vec2 curP = cam->map(curDir);
    if (cam->isOnImage(curP, settingResidualPatternHeight)) {
      pointFound = true;
      alpha0 = alpha;
      break;
    }
  }

  if (!pointFound)
    return;

  std::vector<std::pair<Vec2, double>> energiesFound;
  double bestEnergy = INF;
  Vec2 bestPoint;
  double bestDepth;
  for (int sgn : {-1, 1}) {
    double alpha = alpha0;
    Vec2 point;
    do {
      Vec3 curDir = (1 - alpha) * dirMax + alpha * dirMin;
      ceres::Jet<double, 3> curDirJet[3];
      for (int i = 0; i < 3; ++i) {
        curDirJet[i].a = curDir[i];
        curDirJet[i].v[i] = 1;
      }
      Eigen::Matrix<ceres::Jet<double, 3>, 2, 1> pointJet = cam->map(curDirJet);
      point[0] = pointJet[0].a;
      point[1] = pointJet[1].a;

      curDir.normalize();
      Vec2 depths = triangulate(baseToRef, baseDirections[0], curDir);
      Vec2 reproj[settingResidualPatternSize];
      for (int i = 0; i < settingResidualPatternSize; ++i)
        reproj[i] = cam->map(baseToRef * (depths[0] * baseDirections[i]));

      if (debugOut) {
        cv::Point cvp = toCvPoint(reproj[0]);
        cv::rectangle(curved, cvp - cv::Point(1, 1), cvp + cv::Point(1, 1),
                      CV_GREEN, cv::FILLED);
        cv::imshow("epipolar curve", curved);
        cv::waitKey();
      }

      energiesFound.push_back({reproj[0], INF});

      double maxReprojDist = -1;
      for (int i = 1; i < settingResidualPatternSize; ++i) {
        double dist = (reproj[i] - reproj[0]).norm();
        if (maxReprojDist < dist)
          maxReprojDist = dist;
      }
      int pyrLevel =
          std::round(std::log2(maxReprojDist / settingResidualPatternHeight));
      if (pyrLevel < 0)
        pyrLevel = 0;
      if (pyrLevel >= settingPyrLevels)
        pyrLevel = settingPyrLevels - 1;
      for (Vec2 &r : reproj)
        r /= static_cast<double>(1 << pyrLevel);

      double energy = 0;
      for (int i = 0; i < settingResidualPatternSize; ++i) {
        double residual = std::abs(
            baseIntencities[i] -
            lightRefToBase(refFrame.framePyr[pyrLevel](toCvPoint(reproj[i]))));
        energy += residual > settingEpipolarOutlierIntencityDiff
                      ? 2 * residual - 1
                      : residual * residual;
      }

      energiesFound.back().second = energy;

      if (energy < bestEnergy) {
        bestEnergy = energy;
        bestPoint = point;
      }

      Mat23 mapJacobian;
      mapJacobian << pointJet[0].v.transpose(), pointJet[1].v.transpose();
      double deltaAlpha = 1. / (mapJacobian * (dirMax - dirMin)).norm();
      alpha += sgn * deltaAlpha;
    } while (cam->isOnImage(point, settingResidualPatternHeight));
  } 

  minDepth = bestDepth - 0.5;
  maxDepth = bestDepth + 0.5;
  depth = bestDepth;

  double secondBestEnergy = INF;
  for (const auto &p : energiesFound) {
    if ((p.first - bestPoint).norm() < settingMinSecondBestDistance)
      continue;
    if (p.second < secondBestEnergy)
      secondBestEnergy = p.second;
  }

  quality = secondBestEnergy / bestEnergy;
}

} // namespace fishdso
