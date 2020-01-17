#include "system/ImmaturePoint.h"
#include "PreKeyFrameEntryInternals.h"
#include "system/KeyFrame.h"
#include "util/defs.h"
#include "util/geometry.h"
#include "util/util.h"
#include <ceres/internal/autodiff.h>
#include <ceres/jet.h>

namespace mdso {

#define PL (settings.pyramid.levelNum())
#define PS (settings.residualPattern.pattern().size())
#define PH (settings.residualPattern.height)
#define TH (settings.intensity.outlierDiff)

std::string ImmaturePoint::statusName(ImmaturePoint::TracingStatus status) {
  switch (status) {
  case OK:
    return "OK";
    break;
  case WAS_OOB:
    return "WAS_OOB";
    break;
  case BIG_PREDICTED_ERROR:
    return "BIG_PREDICTED_ERROR";
    break;
  case EPIPOLAR_OOB:
    return "EPIPOLAR_OOB";
    break;
  case TOO_COARSE_PYR_LEVEL:
    return "TOO_COARSE_PYR_LEVEL";
    break;
  case INF_DEPTH:
    return "INF_DEPTH";
    break;
  case INF_ENERGY:
    return "INF_ENERGY";
    break;
  case BIG_ENERGY:
    return "BIG_ENERGY";
    break;
  case SMALL_ABS_SECOND_BEST:
    return "SMALL_ABS_SECOND_BEST";
    break;
  case LOW_QUALITY:
    return "LOW_QUALITY";
    break;
  case STATUS_COUNT:
    return "STATUS_COUNT";
    break;
  }

  CHECK(false);
  return "Unexpected";
}

ImmaturePoint::ImmaturePoint(KeyFrameEntry *host, const Vec2 &p,
                             const PointTracerSettings &settings)
    : p(p)
    , minDepth(0)
    , maxDepth(INF)
    , bestQuality(-1)
    , lastEnergy(INF)
    , stddev(INF)
    , state(ACTIVE)
    , host(host)
    , mIsReady(false)
    , lastTraced(false)
    , numTraced(0)
    , tracedPyrLevel(0) {
  CHECK(host);
  const PreKeyFrame &baseFrame = *host->host->preKeyFrame;

  cam = baseFrame.cam;

  camBase = &cam->bundle[host->ind].cam;
  if (!camBase->isOnImage(p, PH)) {
    state = OOB;
    return;
  }

  for (int i = 0; i < PS; ++i) {
    Vec2 curP = p + settings.residualPattern.pattern()[i];
    cv::Point curPCV = toCvPoint(curP);
    baseDirections[i] = camBase->unmap(curP).normalized();
    baseIntencities[i] = baseFrame.image(host->ind)(curPCV);

    baseGrad[i] = Vec2(baseFrame.frames[host->ind].gradX(curPCV),
                       baseFrame.frames[host->ind].gradY(curPCV));
    baseGradNorm[i] = baseGrad[i].normalized();
  }

  dir = baseDirections[0];
}

bool ImmaturePoint::isReady() const { return mIsReady; }

void ImmaturePoint::setTrueDepth(double trueDepth,
                                 const Settings::PointTracer &ptSettings) {
  depth = trueDepth;
  stddev = ptSettings.optimizedStddev / 2;
  double delta = depth * ptSettings.relTrueDepthDelta;
  minDepth = depth - delta;
  maxDepth = depth + delta;
  bestQuality = INF;
  lastEnergy = 0;
  state = ACTIVE;
}

int ImmaturePoint::pointsToTrace(const CameraModel &camRef,
                                 const SE3 &baseToRef, Vec3 &dirMinDepth,
                                 Vec3 &dirMaxDepth, Vec2 points[],
                                 Vec3 directions[],
                                 const PointTracerSettings &settings) {
  if (!camRef.isOnImage(camRef.map(dirMaxDepth), PH))
    return 0;

  int maxSearchCount = std::min(int(settings.pointTracer.maxSearchRel *
                                    (camRef.getWidth() + camRef.getHeight())),
                                settings.pointTracer.maxSearchAbs());
  int size = 0;

  double alpha = 0;
  Vec2 point;
  int pointCnt = 0;
  do {
    Vec3 curDir = (1 - alpha) * dirMaxDepth + alpha * dirMinDepth;
    Mat23 mapJacobian;
    std::tie(point, mapJacobian) = camRef.diffMap(curDir);

    points[size] = point;
    directions[size] = curDir;
    size++;

    double deltaAlpha = 1. / (mapJacobian * (dirMaxDepth - dirMinDepth)).norm();
    alpha += deltaAlpha;

    pointCnt++;
  } while (pointCnt < maxSearchCount && camRef.isOnImage(point, PH) &&
           alpha <= 1);

  return size;
}

Vec2 ImmaturePoint::tracePrecise(
    const ceres::BiCubicInterpolator<ceres::Grid2D<unsigned char, 1>> &refFrame,
    const Vec2 &from, const Vec2 &to, double intencities[], Vec2 pattern[],
    double &bestDispl, double &bestEnergy,
    const PointTracerSettings &settings) {
  Vec2 dir = to - from;
  dir.normalize();
  Vec2 bestPoint = (from + to) * 0.5;
  bestEnergy = INF;
  bestDispl = 0;
  double step = 0;
  for (int it = 0; it < settings.pointTracer.gnIter + 1; ++it) {
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
      if (settings.pointTracer.useAltHWeighting) {
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

double ImmaturePoint::estVariance(const Vec2 &searchDirection,
                                  const PointTracerSettings &settings) {
  double sum1 = 0;
  for (const Vec2 &gN : baseGradNorm) {
    double s = gN.dot(searchDirection);
    sum1 += s * s;
  }

  lastVar = PS * settings.pointTracer.positionVariance / sum1;
  return lastVar;
}

ImmaturePoint::TracingStatus
ImmaturePoint::traceOn(const PreKeyFrame::FrameEntry &refFrameEntry,
                       TracingDebugType debugType,
                       const PointTracerSettings &settings) {
  if (state == OOB)
    return WAS_OOB;

  KeyFrame &baseFrame = *host->host;
  PreKeyFrame &refFrame = *refFrameEntry.host;
  int indBase = host->ind;
  int indRef = refFrameEntry.ind;
  CameraModel &camRef = cam->bundle[indRef].cam;

  AffLight lightBaseToRef =
      refFrameEntry.lightBaseToThis *
      refFrame.baseFrame->frames[indRef].lightWorldToThis *
      host->lightWorldToThis.inverse();
  SE3 baseToRef = cam->bundle[indRef].bodyToThis * refFrame.baseToThis() *
                  refFrame.baseFrame->thisToWorld().inverse() *
                  baseFrame.thisToWorld() *
                  cam->bundle[indBase].bodyToThis.inverse();

  Vec3 dirMin = (baseToRef * (minDepth * baseDirections[0])).normalized();
  Vec3 dirMax = maxDepth == INF
                    ? baseToRef.so3() * baseDirections[0]
                    : (baseToRef * (maxDepth * baseDirections[0])).normalized();
  Mat23 jacobian = cam->bundle[indRef].cam.diffMap(dirMax).second;
  Vec2 searchDirection = jacobian * (dirMin - dirMax);
  searchDirection.normalize();

  double variance = estVariance(searchDirection, settings);
  double curDev = std::sqrt(variance);

  if (numTraced > 0)
    if (curDev * settings.pointTracer.imprFactor > stddev)
      return BIG_PREDICTED_ERROR;

  static_vector<Vec2, Settings::PointTracer::max_maxSearchAbs> points(
      Settings::PointTracer::max_maxSearchAbs);
  static_vector<Vec3, Settings::PointTracer::max_maxSearchAbs> directions(
      Settings::PointTracer::max_maxSearchAbs);
  int size = pointsToTrace(camRef, baseToRef, dirMin, dirMax, points.data(),
                           directions.data(), settings);
  if (size == 0)
    return EPIPOLAR_OOB;
  points.resize(size);
  directions.resize(size);

  static_vector<double, Settings::ResidualPattern::max_size> intencities(PS);
  for (int i = 0; i < PS; ++i)
    intencities[i] = lightBaseToRef(baseIntencities[i]);

  static_vector<std::pair<Vec2, double>,
                Settings::PointTracer::max_maxSearchAbs>
      energiesFound;
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
    static_vector<Vec2, Settings::ResidualPattern::max_size> reproj(PS);
    reproj[0] = point;
    double curDepth = INF;
    if (maxDepth == INF && dirInd == 0) {
      for (int i = 1; i < PS; ++i)
        reproj[i] = camRef.map(baseToRef.so3() * baseDirections[i]);
    } else {
      Vec2 curDepths = triangulate(baseToRef, baseDirections[0], curDir);
      curDepth = curDepths[0];
      for (int i = 1; i < PS; ++i)
        reproj[i] = camRef.map(baseToRef * (curDepths[0] * baseDirections[i]));
    }

    double maxReprojDist = -1;
    for (int i = 1; i < PS; ++i) {
      double dist = (reproj[i] - point).norm();
      if (maxReprojDist < dist)
        maxReprojDist = dist;
    }
    int pyrLevel = std::lround(std::log2(maxReprojDist / PH));
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
      refFrame.frames[indRef].internals->interpolator(pyrLevel).Evaluate(
          reproj[i][1], reproj[i][0], &refIntencity);
      double residual = std::abs(intencities[i] - refIntencity);
      energy += residual > TH ? TH * (2 * residual - TH) : residual * residual;
    }

    energiesFound.push_back({point, energy});

    if (energy < bestEnergy) {
      bestEnergy = energy;
      bestPoint = point;
      bestDepth = curDepth;
      bestInd = dirInd;
      bestPyrLevel = pyrLevel;
    }
  }

  lastEnergy = bestEnergy;

  lastTraced = false;

  // tracing reliability checks
  if (bestPyrLevel == PL - 1)
    return TOO_COARSE_PYR_LEVEL;

  if (bestDepth == INF)
    return INF_DEPTH;

  double secondBestEnergy = INF;
  for (const auto &[p, e] : energiesFound) {
    if ((p - bestPoint).norm() < settings.pointTracer.minSecondBestDistance)
      continue;
    if (e < secondBestEnergy)
      secondBestEnergy = e;
  }

  if (bestEnergy == INF || secondBestEnergy == INF)
    return INF_ENERGY;

  double secondBestEnergyThres =
      settings.pointTracer.secondBestEnergyThresFactor * PS * TH * TH;
  if (secondBestEnergy <= secondBestEnergyThres)
    return SMALL_ABS_SECOND_BEST;

  double outlierEnergy =
      settings.pointTracer.outlierEnergyFactor * PS * TH * TH;

  if (lastEnergy > outlierEnergy)
    return BIG_ENERGY;

  double newQuality = secondBestEnergy / bestEnergy;

  if (newQuality < settings.pointTracer.outlierQuality)
    return LOW_QUALITY;

  if (newQuality > bestQuality)
    bestQuality = newQuality;

  numTraced++;

  eBeforeSubpixel = bestEnergy;
  depthBeforeSubpixel = bestDepth;

  // subpixel refinement
  double bestDispl = 0;
  if (settings.pointTracer.gnIter > 0) {
    int fromInd = std::max(0, bestInd - 1);
    int toInd = std::min(int(points.size()) - 1, bestInd + 1);
    Vec2 from = points[fromInd];
    Vec2 to = points[toInd];
    static_vector<Vec2, Settings::ResidualPattern::max_size> pattern(PS);
    double scale = 1.0 / (1 << bestPyrLevel);
    pattern[0] = Vec2::Zero();
    for (int i = 1; i < PS; ++i) {
      Vec2 reproj = camRef.map(baseToRef * (bestDepth * baseDirections[i]));
      pattern[i] = scale * (reproj - points[bestInd]);
    }
    bestPoint = tracePrecise(
        refFrame.frames[indRef].internals->interpolator(bestPyrLevel), from, to,
        intencities.data(), pattern.data(), bestDispl, bestEnergy, settings);
    depth = triangulate(baseToRef, baseDirections[0],
                        camRef.unmap((bestPoint / scale).eval()))[0];
  } else
    depth = bestDepth;

  eAfterSubpixel = bestEnergy;

  // depth bounds
  double displ = 2 * curDev;
  Vec2 minDepthPos =
      approxOnCurve(points.data(), points.size(), bestInd + displ);
  minDepth =
      triangulate(baseToRef, baseDirections[0], camRef.unmap(minDepthPos))[0];

  double maxDepthDispl = bestInd - displ;
  if (maxDepthDispl <= 0)
    maxDepth = INF;
  else {
    Vec2 maxDepthPos =
        approxOnCurve(points.data(), points.size(), maxDepthDispl);
    maxDepth =
        triangulate(baseToRef, baseDirections[0], camRef.unmap(maxDepthPos))[0];
  }

  lastTraced = true;
  stddev = curDev;
  if (stddev < settings.pointTracer.optimizedStddev)
    mIsReady = true;

  tracedPyrLevel = bestPyrLevel;

  if (state == ACTIVE && debugType == DRAW_EPIPOLE) {
    cv::Mat base;
    cv::Mat curved;
    base = baseFrame.preKeyFrame->frames[indBase].frameColored.clone();
    curved = refFrame.frames[indRef].frameColored.clone();

    cv::circle(base, toCvPoint(p), 4, CV_GREEN, 1);
    drawTracing(curved, energiesFound.data(), energiesFound.size(), 5);

    int wantedHeight = 600;
    int wantedWidth =
        double(camRef.getWidth()) / camRef.getHeight() * wantedHeight;
    double fx = double(wantedWidth) / camRef.getWidth();
    double fy = double(wantedHeight) / camRef.getHeight();
    cv::Mat curv2;
    cv::resize(curved, curv2, cv::Size(), fx, fy);
    cv::imshow("epipolar curve", curv2);

    cv::Mat base2;
    cv::resize(base, base2, cv::Size(), fx, fy);
    cv::imshow("base frame", base2);

    cv::waitKey();
  }

  return OK;
}

void ImmaturePoint::drawTracing(cv::Mat &frame,
                                std::pair<Vec2, double> energiesFound[],
                                int size, int lineWidth) {
  auto comp = [](const std::pair<Vec2, double> &a,
                 const std::pair<Vec2, double> &b) {
    return a.second < b.second;
  };
  int minI = std::min_element(energiesFound, energiesFound + size, comp) -
             energiesFound;
  double minEnergy = energiesFound[minI].second;

  double maxEnergy =
      std::max_element(energiesFound, energiesFound, comp)->second;

  for (int i = 0; i < size; ++i) {
    Vec2 p = energiesFound[i].first;
    double e = energiesFound[i].second;
    Vec2 dir = i == size - 1 ? p - energiesFound[i - 1].first
                             : energiesFound[i + 1].first - p;
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

} // namespace mdso
