#include "system/ImmaturePoint.h"
#include "util/defs.h"
#include "util/geometry.h"
#include "util/util.h"
#include <ceres/internal/autodiff.h>
#include <ceres/jet.h>

namespace fishdso {

#define PL (pyrSettings.levelNum)
#define PS (rpSettings.pattern().size())
#define PH (rpSettings.height)
#define TH (intencitySettings.outlierDiff)

ImmaturePoint::ImmaturePoint(PreKeyFrame *baseFrame, const Vec2 &p,
                             const Settings::PointTracer &tracingSettings,
                             const Settings::Intencity &intencitySettings,
                             const Settings::ResidualPattern &rpSettings,
                             const Settings::Pyramid &pyrSettings)
    : p(p)
    , baseDirections(rpSettings.pattern().size())
    , baseIntencities(rpSettings.pattern().size())
    , baseGrad(rpSettings.pattern().size())
    , baseGradNorm(rpSettings.pattern().size())
    , minDepth(0)
    , maxDepth(INF)
    , bestQuality(-1)
    , lastEnergy(INF)
    , stddev(INF)
    , baseFrame(baseFrame)
    , cam(baseFrame->cam)
    , state(ACTIVE)
    , tracingSettings(tracingSettings)
    , intencitySettings(intencitySettings)
    , rpSettings(rpSettings)
    , pyrSettings(pyrSettings)
    , lastTraced(false)
    , numTraced(0)
    , tracedPyrLevel(0) {
  if (!cam->isOnImage(p, PH)) {
    state = OOB;
    return;
  }

  for (int i = 0; i < PS; ++i) {
    Vec2 curP = p + rpSettings.pattern()[i];
    cv::Point curPCV = toCvPoint(curP);
    baseDirections[i] = cam->unmap(curP).normalized();
    baseIntencities[i] = baseFrame->frame()(curPCV);

    baseGrad[i] = Vec2(baseFrame->gradX(curPCV), baseFrame->gradY(curPCV));
    baseGradNorm[i] = baseGrad[i].normalized();
  }
}

bool ImmaturePoint::isReady() {
  return state == ACTIVE && stddev < tracingSettings.optimizedStddev;
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

  if (tracingSettings.performFullTracing) {
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
      tracingSettings.maxSearchRel * (cam->getWidth() + cam->getHeight());
  double alpha0 = 0;
  double step = 1.0 / (tracingSettings.onImageTestCount - 1);
  while (alpha0 <= 1) {
    Vec3 curDir = (1 - alpha0) * dirMaxDepth + alpha0 * dirMinDepth;
    Vec2 curP = cam->map(curDir);
    if (!cam->isOnImage(curP, PH)) {
      if (!tracingSettings.performFullTracing)
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
        if (!tracingSettings.performFullTracing && pointCnt >= maxSearchCount)
          break;
      } while (alpha >= 0 && alpha <= 1 && cam->isOnImage(point, PH));
    }

    if (!tracingSettings.performFullTracing)
      break;

    alpha0 = alpha + step;
  }

  return points.size() > 1;
}

Vec2 ImmaturePoint::tracePrecise(
    const ceres::BiCubicInterpolator<ceres::Grid2D<unsigned char, 1>> &refFrame,
    const Vec2 &from, const Vec2 &to, const std::vector<double> &intencities,
    const StdVector<Vec2> &pattern, double &bestDispl, double &bestEnergy) {
  Vec2 dir = to - from;
  dir.normalize();
  Vec2 bestPoint = (from + to) * 0.5;
  bestEnergy = INF;
  bestDispl = 0;
  double step = 0;
  for (int it = 0; it < tracingSettings.gnIter + 1; ++it) {
    double newEnergy = 0;
    double H = 0, b = 0;
    Vec2 curPoint = bestPoint + step * dir;
    for (int i = 0; i < PS; ++i) {
      double intencity;
      Vec2 p = curPoint + pattern[i];
      Vec2 grad;
      refFrame.Evaluate(p[1], p[0], &intencity, &grad[1], &grad[0]);
      double r = intencity - intencities[i];
      double ar = std::abs(r);
      double wb = ar > TH ? TH / ar : 1;
      newEnergy += wb * (2 - wb) * ar * ar;
      double dr = grad.dot(dir);
      b += wb * r * dr;
      if (tracingSettings.useAltHWeighting) {
        double wh = wb / (2 - wb);
        H += wh * dr * dr;
      } else
        H += wb * dr * dr;
    }

    if (newEnergy < bestEnergy) {
      bestEnergy = newEnergy;
      bestPoint = curPoint;
      bestDispl += step;
      step = -b / H;
      if (bestDispl + step > 1)
        step = (1 - bestDispl) * 0.5;
      else if (bestDispl + step < -1)
        step = (-1 - bestDispl) * 0.5;
    } else
      step *= 0.5;
  }

  return bestPoint;
}

double ImmaturePoint::estVariance(const Vec2 &searchDirection) {
  double sum1 = 0;
  for (const Vec2 &gN : baseGradNorm) {
    double s = gN.dot(searchDirection);
    sum1 += s * s;
  }

  lastGeomVar = PS * tracingSettings.positionVariance / sum1;
  return lastGeomVar;
}

void ImmaturePoint::traceOn(const PreKeyFrame &refFrame,
                            TracingDebugType debugType) {
  if (state == OOB)
    return;

  double oldDepth = depth;

  AffineLightTransform<double> lightBaseToRef =
      refFrame.lightWorldToThis * baseFrame->lightWorldToThis.inverse();
  SE3 baseToRef = refFrame.worldToThis * baseFrame->worldToThis.inverse();

  Vec3 dirMin = (baseToRef * (minDepth * baseDirections[0])).normalized();
  Vec3 dirMax = maxDepth == INF
                    ? baseToRef.so3() * baseDirections[0]
                    : (baseToRef * (maxDepth * baseDirections[0])).normalized();
  Mat23 jacobian = cam->diffMap(dirMax).second;
  Vec2 searchDirection = jacobian * (dirMin - dirMax);
  searchDirection.normalize();

  double variance = estVariance(searchDirection);
  double curDev = std::sqrt(variance);

  if (!tracingSettings.performFullTracing && numTraced > 0)
    if (curDev * tracingSettings.imprFactor > stddev)
      return;

  StdVector<Vec2> points;
  std::vector<Vec3> directions;
  if (!pointsToTrace(baseToRef, dirMin, dirMax, points, directions)) {
    return;
  }

  std::vector<double> intencities(PS);
  for (int i = 0; i < PS; ++i)
    intencities[i] = lightBaseToRef(baseIntencities[i]);

  StdVector<std::pair<Vec2, double>> energiesFound;
  double bestEnergy = INF;
  Vec2 bestPoint;
  double bestDepth;
  int bestInd;
  pyrChanged = false;
  int bestPyrLevel = -1;
  int lastPyrLevel = -1;

  for (int dirInd = 0; dirInd < directions.size(); ++dirInd) {
    Vec3 curDir = directions[dirInd];
    Vec2 point = points[dirInd];
    curDir.normalize();
    StdVector<Vec2> reproj(PS);
    reproj[0] = point;
    double depth = INF;
    if (maxDepth == INF && dirInd == 0) {
      for (int i = 1; i < PS; ++i)
        reproj[i] = cam->map(baseToRef.so3() * baseDirections[i]);
    } else {
      Vec2 depths = triangulate(baseToRef, baseDirections[0], curDir);
      depth = depths[0];
      for (int i = 1; i < PS; ++i)
        reproj[i] = cam->map(baseToRef * (depths[0] * baseDirections[i]));
    }

    double maxReprojDist = -1;
    for (int i = 1; i < PS; ++i) {
      double dist = (reproj[i] - point).norm();
      if (maxReprojDist < dist)
        maxReprojDist = dist;
    }
    int pyrLevel = std::round(std::log2(maxReprojDist / PH));
    if (pyrLevel < 0)
      pyrLevel = 0;
    if (pyrLevel >= PL)
      pyrLevel = PL - 1;

    if (lastPyrLevel != -1 && lastPyrLevel != pyrLevel)
      pyrChanged = true;

    lastPyrLevel = pyrLevel;

    for (Vec2 &r : reproj)
      r /= double(1 << pyrLevel);

    double energy = 0;
    for (int i = 0; i < PS; ++i) {
      double refIntencity;
      refFrame.framePyr.interpolator(pyrLevel).Evaluate(
          reproj[i][1], reproj[i][0], &refIntencity);
      double residual = std::abs(intencities[i] - refIntencity);
      energy += residual > TH ? TH * (2 * residual - TH) : residual * residual;
    }

    energiesFound.push_back({point, energy});

    if (energy < bestEnergy) {
      bestEnergy = energy;
      bestPoint = point;
      bestDepth = depth;
      bestInd = dirInd;
      bestPyrLevel = pyrLevel;
    }
  }

  lastTraced = false;

  // tracing reliability checks
  if (bestPyrLevel == PL - 1) {
    depth = oldDepth;
    return;
  }

  if (bestDepth == INF) {
    depth = oldDepth;
    LOG(INFO) << "trace quit : inf point\n";
    return;
  }

  double secondBestEnergy = INF;
  for (const auto &p : energiesFound) {
    if ((p.first - bestPoint).norm() < tracingSettings.minSecondBestDistance)
      continue;
    if (p.second < secondBestEnergy)
      secondBestEnergy = p.second;
  }

  if (bestEnergy == INF || secondBestEnergy == INF || secondBestEnergy <= 5.0) {
    LOG(INFO) << "trace quit : energies\n";
    depth = oldDepth;
    return;
  }

  double newQuality = secondBestEnergy / bestEnergy;
  if (newQuality > bestQuality)
    bestQuality = newQuality;
  lastEnergy = bestEnergy;

  double outlierEnergy = tracingSettings.outlierEnergyFactor * PS * TH * TH;
  if (lastEnergy > outlierEnergy ||
      newQuality < tracingSettings.outlierQuality) {
    depth = oldDepth;
    LOG(INFO) << "trace quit : e or q\n";
    return;
  }

  numTraced++;

  eBeforeSubpixel = bestEnergy;
  depthBeforeSubpixel =
      triangulate(baseToRef, baseDirections[0], cam->unmap(bestPoint))[0];

  // subpixel refinement
  double bestDispl = 0;
  if (tracingSettings.gnIter > 0) {
    int fromInd = std::max(0, bestInd - 1);
    int toInd = std::min(int(points.size()) - 1, bestInd + 1);
    Vec2 from = points[fromInd];
    Vec2 to = points[toInd];
    StdVector<Vec2> pattern(PS);
    double scale = 1.0 / (1 << bestPyrLevel);
    pattern[0] = Vec2::Zero();
    for (int i = 1; i < PS; ++i) {
      Vec2 reproj = cam->map(baseToRef * (bestDepth * baseDirections[i]));
      pattern[i] = scale * (reproj - points[bestInd]);
    }
    bestPoint = tracePrecise(refFrame.framePyr.interpolator(bestPyrLevel), from,
                             to, intencities, pattern, bestDispl, bestEnergy);
    depth = triangulate(baseToRef, baseDirections[0],
                        cam->unmap(bestPoint / scale))[0];
  } else
    depth = bestDepth;

  eAfterSubpixel = bestEnergy;

  double displ = 2 * curDev;
  // depth bounds
  Vec2 minDepthPos = approxOnCurve(points, bestInd + displ);
  minDepth =
      triangulate(baseToRef, baseDirections[0], cam->unmap(minDepthPos))[0];

  double maxDepthDispl = bestInd - displ;
  if (maxDepthDispl <= 0)
    maxDepth = INF;
  else {
    Vec2 maxDepthPos = approxOnCurve(points, maxDepthDispl);
    maxDepth =
        triangulate(baseToRef, baseDirections[0], cam->unmap(maxDepthPos))[0];
  }

  lastTraced = true;
  stddev = curDev;
  tracedPyrLevel = bestPyrLevel;

  if (state == ACTIVE && debugType == DRAW_EPIPOLE) {
    cv::Mat base;
    cv::Mat curved;
    base = baseFrame->frameColored.clone();
    curved = refFrame.frameColored.clone();

    cv::circle(base, toCvPoint(p), 4, CV_GREEN, 1);
    drawTracing(curved, energiesFound, 5);

    int wantedHeight = 600;
    int wantedWidth = double(cam->getWidth()) / cam->getHeight() * wantedHeight;
    double fx = double(wantedWidth) / cam->getWidth();
    double fy = double(wantedHeight) / cam->getHeight();
    cv::Mat curv2;
    cv::resize(curved, curv2, cv::Size(), fx, fy);
    cv::imshow("epipolar curve", curv2);

    cv::Mat base2;
    cv::resize(base, base2, cv::Size(), fx, fy);
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
