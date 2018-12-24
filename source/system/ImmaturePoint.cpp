#include "system/ImmaturePoint.h"
#include "util/defs.h"
#include "util/geometry.h"
#include "util/util.h"
#include <ceres/internal/autodiff.h>
#include <ceres/jet.h>

namespace fishdso {

ImmaturePoint::ImmaturePoint(PreKeyFrame *baseFrame, const Vec2 &p)
    : p(p), minDepth(0), maxDepth(INF), quality(-1), baseFrame(baseFrame),
      cam(baseFrame->cam), state(ACTIVE) {
  if (!cam->isOnImage(p, settingResidualPatternHeight)) {
    state = OOB;
    return;
  }
  for (int i = 0; i < settingResidualPatternSize; ++i) {
    baseDirections[i] = cam->unmap(p + settingResidualPattern[i]).normalized();
    baseIntencities[i] =
        baseFrame->frame()(toCvPoint(p + settingResidualPattern[i]));
  }
}

void ImmaturePoint::traceOn(const PreKeyFrame &refFrame,
                            TracingDebugType debugType) {
  AffineLightTransform<double> lightRefToBase =
      baseFrame->lightWorldToThis * refFrame.lightWorldToThis.inverse();
  SE3 baseToRef = refFrame.worldToThis * baseFrame->worldToThis.inverse();
  Vec3 dirMin = (baseToRef * (minDepth * baseDirections[0])).normalized();
  Vec3 dirMax = maxDepth == INF
                    ? baseToRef.so3() * baseDirections[0]
                    : (baseToRef * (maxDepth * baseDirections[0])).normalized();
  if (M_PI - angle(dirMin, dirMax) < 1e-3)
    return;

  // While searching along epipolar curve, we will continously map rays on a
  // diametrical segment of a sphere. Since our camera model remains valid only
  // when angle between the mapped ray and Oz is smaller then certain maxAngle,
  // we want to intersect the segment of search with the "well-mapped" part of
  // the sphere, i.e. z > z0.
  if (!intersectOnSphere(cam->getMaxAngle(), dirMin, dirMax))
    return;

  StdVector<std::pair<Vec2, double>> energiesFound;
  double bestEnergy = INF;
  Vec2 bestPoint;
  double bestDepth;

  double step = 1.0 / (settingEpipolarOnImageTestCount - 1);
  double alpha0 = 0;
  while (alpha0 <= 1) {
    Vec3 curDir = (1 - alpha0) * dirMax + alpha0 * dirMin;
    Vec2 curP = cam->map(curDir);
    if (!cam->isOnImage(curP, settingResidualPatternHeight)) {
      alpha0 += step;
      continue;
    }

    double alpha;
    for (int sgn : {-1, 1}) {
      alpha = alpha0;
      Vec2 point;
      do {
        Vec3 curDir = (1 - alpha) * dirMax + alpha * dirMin;
        ceres::Jet<double, 3> curDirJet[3];
        for (int i = 0; i < 3; ++i) {
          curDirJet[i].a = curDir[i];
          curDirJet[i].v[i] = 1;
        }
        Eigen::Matrix<ceres::Jet<double, 3>, 2, 1> pointJet =
            cam->map(curDirJet);
        point[0] = pointJet[0].a;
        point[1] = pointJet[1].a;

        curDir.normalize();
        Vec2 depths = triangulate(baseToRef, baseDirections[0], curDir);
        Vec2 reproj[settingResidualPatternSize];
        reproj[0] = point;
        for (int i = 1; i < settingResidualPatternSize; ++i)
          reproj[i] = cam->map(baseToRef * (depths[0] * baseDirections[i]));

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
          double residual = std::abs(baseIntencities[i] -
                                     lightRefToBase(refFrame.framePyr[pyrLevel](
                                         toCvPoint(reproj[i]))));
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
      } while (alpha >= 0 && alpha <= 1 &&
               cam->isOnImage(point, settingResidualPatternHeight));
    }

    alpha0 = alpha + step;
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

  if (debugType == DRAW_EPIPOLE) {
    cv::Mat base;
    cv::Mat curved;
    base = baseFrame->frameColored.clone();
    curved = refFrame.frameColored.clone();

    cv::circle(base, toCvPoint(p), 10, CV_GREEN, cv::FILLED);
    drawTracing(curved, energiesFound, 8);

    cv::Mat curv2;
    cv::resize(curved, curv2, cv::Size(), 0.5, 0.5);
    cv::imshow("epipolar curve", curv2);

    cv::Mat base2;
    cv::resize(base, base2, cv::Size(), 0.5, 0.5);
    cv::imshow("base frame", base2);

    cv::waitKey();
  }
}

void ImmaturePoint::drawTracing(
    cv::Mat &frame, const StdVector<std::pair<Vec2, double>> &energiesFound,
    int lineWidth) {
  auto comp = [](const std::pair<Vec2, double> &a,
                 const std::pair<Vec2, double> &b) {
    return a.second < b.second;
  };
  int minI =
      std::min_element(energiesFound.begin(), energiesFound.end(), comp) -
      energiesFound.begin();
  double minEnergy = energiesFound[minI].second;

  double maxEnergy =
      std::max_element(energiesFound.begin(), energiesFound.end(), comp)
          ->second;

  for (int i = 0; i < energiesFound.size(); ++i) {
    double e = energiesFound[i].second;
    Vec2 p = energiesFound[i].first;
    Vec2 dir = (i == energiesFound.size() - 1 ? p - energiesFound[i - 1].first
                                              : energiesFound[i + 1].first - p);
    Vec2 ort(dir[1], -dir[0]);
    ort.normalize();
    if (i == minI)
      ort *= 2;
    Vec2 beg = p + ort * (double(lineWidth) / 2);
    Vec2 end = p - ort * (double(lineWidth) / 2);

    cv::Scalar color =
        (i == minI ? CV_BLACK : depthCol(e, minEnergy, maxEnergy));
    int thickness = (i == minI ? 3 : 1);
    cv::line(frame, toCvPoint(beg), toCvPoint(end), color, thickness);
  }
}

} // namespace fishdso
