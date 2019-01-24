#include "system/ImmaturePoint.h"
#include "util/defs.h"
#include "util/geometry.h"
#include "util/util.h"
#include <ceres/internal/autodiff.h>
#include <ceres/jet.h>

namespace fishdso {

ImmaturePoint::ImmaturePoint(PreKeyFrame *baseFrame, const Vec2 &p)
    : p(p), minDepth(0), maxDepth(INF), bestQuality(-1), lastEnergy(INF),
      baseFrame(baseFrame), cam(baseFrame->cam), state(ACTIVE) {
  if (!cam->isOnImage(p, settingResidualPatternHeight)) {
    state = OOB;
    return;
  }
  for (int i = 0; i < settingResidualPatternSize; ++i) {
    Vec2 curP = p + settingResidualPattern[i];
    cv::Point curPCV = toCvPoint(curP);
    baseDirections[i] = cam->unmap(curP).normalized();
    baseIntencities[i] = baseFrame->frame()(curPCV);

    baseGrad[i] = Vec2(baseFrame->gradX(curPCV), baseFrame->gradY(curPCV));
    baseGradNorm[i] = baseGrad[i].normalized();
  }
}

bool ImmaturePoint::pointsToTrace(const SE3 &baseToRef, Vec3 &dirMinDepth,
                                  Vec3 &dirMaxDepth, StdVector<Vec2> &points,
                                  std::vector<Vec3> &directions) {
  points.resize(0);
  directions.resize(0);

  if (M_PI - angle(dirMinDepth, dirMaxDepth) < 1e-3) {
    LOG(INFO) << "ret by angle" << std::endl;
    return false;
  }

  if (FLAGS_perform_full_tracing) {
    // While searching along epipolar curve, we will continously map rays on a
    // diametrical segment of a sphere. Since our camera model remains valid
    // only when angle between the mapped ray and Oz is smaller then certain
    // maxAngle, we want to intersect the segment of search with the
    // "well-mapped" part of the sphere, i.e. z > z0.
    if (!intersectOnSphere(cam->getMaxAngle(), dirMinDepth, dirMaxDepth)) {
      LOG(INFO) << "ret by no itersection" << std::endl;
      return false;
    }
  }

  int maxSearchCount =
      settingEpipolarMaxSearchRel * (cam->getWidth() + cam->getHeight());
  double alpha0 = 0;
  double step = 1.0 / (settingEpipolarOnImageTestCount - 1);
  while (alpha0 <= 1) {
    Vec3 curDir = (1 - alpha0) * dirMaxDepth + alpha0 * dirMinDepth;
    Vec2 curP = cam->map(curDir);
    if (!cam->isOnImage(curP, settingResidualPatternHeight)) {
      if (!FLAGS_perform_full_tracing)
        break;
      alpha0 += step;
      continue;
    }

    double alpha;
    for (int sgn : {-1, 1}) {
      alpha = alpha0;
      Vec2 point;
      int pointCnt = 0;
      do {
        Vec3 curDir = (1 - alpha) * dirMaxDepth + alpha * dirMinDepth;
        Mat23 mapJacobian;
        std::tie(point, mapJacobian) = cam->diffMap(curDir);

        points.push_back(point);
        directions.push_back(curDir);

        double deltaAlpha =
            1. / (mapJacobian * (dirMaxDepth - dirMinDepth)).norm();
        alpha += deltaAlpha;

        pointCnt++;
        if (!FLAGS_perform_full_tracing && pointCnt >= maxSearchCount)
          break;
      } while (alpha >= 0 && alpha <= 1 &&
               cam->isOnImage(point, settingResidualPatternHeight));
    }

    if (!FLAGS_perform_full_tracing)
      break;

    alpha0 = alpha + step;
  }

  return points.size() > 1;
}

double ImmaturePoint::estVariance(const Vec2 &searchDirection) {
  double sum1 = 0;
  for (const Vec2 &gN : baseGradNorm) {
    double s = gN.dot(searchDirection);
    sum1 += s * s;
  }

  // double sum2 = 0;
  // for (const Vec2 &g : baseGrad) {
  // double s = g.dot(searchDirection);
  // sum2 += s * s;
  // }

  lastGeomVar = settingEpipolarPositionVariance / sum1;
  // lastIntVar = settingEpipolarIntencityVariance / sum2;
  lastIntVar = 0;
  lastFullVar = lastGeomVar + lastIntVar;

  return settingEpipolarPositionVariance / sum1;
}

void ImmaturePoint::traceOn(const PreKeyFrame &refFrame,
                            TracingDebugType debugType) {
  if (state != ACTIVE)
    return;

  AffineLightTransform<double> lightRefToBase =
      baseFrame->lightWorldToThis * refFrame.lightWorldToThis.inverse();
  SE3 baseToRef = refFrame.worldToThis * baseFrame->worldToThis.inverse();

  Vec3 dirMin = (baseToRef * (minDepth * baseDirections[0])).normalized();
  Vec3 dirMax = maxDepth == INF
                    ? baseToRef.so3() * baseDirections[0]
                    : (baseToRef * (maxDepth * baseDirections[0])).normalized();
  Vec2 pointMin = cam->map(dirMin);
  Vec2 pointMax = cam->map(dirMax);

  Mat23 jacobian = cam->diffMap(dirMax).second;
  Vec2 searchDirection = jacobian * (dirMin - dirMax);
  searchDirection.normalize();

  double variance = estVariance(searchDirection);
  double dev = 2 * std::sqrt(variance);

  if (!FLAGS_perform_full_tracing && maxDepth != INF) {
    double searchLength = (pointMax - pointMin).norm();
    if (dev * settingEpipolarMinImprovementFactor > searchLength) {
      return;
    }
  }

  StdVector<Vec2> points;
  std::vector<Vec3> directions;
  if (!pointsToTrace(baseToRef, dirMin, dirMax, points, directions)) {
    return;
  }

  StdVector<std::pair<Vec2, double>> energiesFound;
  double bestEnergy = INF;
  Vec2 bestPoint;
  double bestDepth;
  int bestInd;

  for (int dirInd = 0; dirInd < directions.size(); ++dirInd) {
    Vec3 curDir = directions[dirInd];
    Vec2 point = points[dirInd];
    curDir.normalize();
    Vec2 depths = triangulate(baseToRef, baseDirections[0], curDir);
    Vec2 reproj[settingResidualPatternSize];
    reproj[0] = point;
    for (int i = 1; i < settingResidualPatternSize; ++i)
      reproj[i] = cam->map(baseToRef * (depths[0] * baseDirections[i]));

    double maxReprojDist = -1;
    for (int i = 1; i < settingResidualPatternSize; ++i) {
      double dist = (reproj[i] - point).norm();
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
      double refIntencity;
      refFrame.framePyr.interpolator(pyrLevel).Evaluate(
          reproj[i][1], reproj[i][0], &refIntencity);
      double residual = baseIntencities[i] - lightRefToBase(refIntencity);
      energy += residual > settingEpipolarOutlierIntencityDiff
                    ? settingEpipolarOutlierIntencityDiff *
                          (2 * std::abs(residual) -
                           settingEpipolarOutlierIntencityDiff)
                    : residual * residual;
    }

    energiesFound.push_back({point, energy});

    if (energy < bestEnergy) {
      bestEnergy = energy;
      bestPoint = point;
      bestDepth = depths[0];
      bestInd = dirInd;
    }
  }

  double devFloor = std::floor(dev);
  double devFrac = dev - devFloor;

  int minDepthInd = bestInd + devFloor;
  Vec2 minDepthPos = points[minDepthInd];
  if (minDepthInd >= points.size() - 1) {
    minDepthInd = points.size() - 1;
    minDepthPos += devFrac * (points[minDepthInd] - points[minDepthInd - 1]);
  } else
    minDepthPos += devFrac * (points[minDepthInd + 1] - points[minDepthInd]);
  Vec2 depthsTemp1 =
      triangulate(baseToRef, baseDirections[0], cam->unmap(minDepthPos));
  minDepth = depthsTemp1[0];

  int maxDepthInd = bestInd - devFloor;
  Vec2 maxDelta;
  if (maxDepthInd <= 0)
    maxDepth = INF;
  else {
    Vec2 maxDepthPos =
        points[maxDepthInd] +
        devFrac * (points[maxDepthInd - 1] - points[maxDepthInd]);
    Vec2 depthTemp2 =
        triangulate(baseToRef, baseDirections[0], cam->unmap(maxDepthPos));
    maxDepth = depthTemp2[0];
  }

  depth = bestDepth;

  double secondBestEnergy = INF;
  for (const auto &p : energiesFound) {
    if ((p.first - bestPoint).norm() < settingMinSecondBestDistance)
      continue;
    if (p.second < secondBestEnergy)
      secondBestEnergy = p.second;
  }

  double newQuality = secondBestEnergy / bestEnergy;
  if (newQuality > bestQuality)
    bestQuality = newQuality;
  lastEnergy = bestEnergy;

  if (bestEnergy == INF || secondBestEnergy == INF || secondBestEnergy <= 1.0)
    state = OUTLIER;

  if (lastEnergy > settingOutlierEpipolarEnergy ||
      bestQuality < settingOutlierEpipolarQuality)
    state = OUTLIER;

  if (debugType == DRAW_EPIPOLE) {
    cv::Mat base;
    cv::Mat curved;
    base = baseFrame->frameColored.clone();
    curved = refFrame.frameColored.clone();

    cv::circle(base, toCvPoint(p), 10, CV_GREEN, 2);
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
