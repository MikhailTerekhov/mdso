#include "system/DsoSystem.h"
#include "output/DsoObserver.h"
#include "output/FrameTrackerObserver.h"
#include "system/AffineLightTransform.h"
#include "system/DelaunayDsoInitializer.h"
#include "system/StereoMatcher.h"
#include "util/defs.h"
#include "util/geometry.h"
#include "util/settings.h"
#include <glog/logging.h>
#include <optional>

namespace fishdso {

DsoSystem::DsoSystem(CameraBundle *cam, const Observers &observers,
                     const Settings &_settings)
    : cam(cam)
    , camPyr(cam->camPyr(_settings.pyramid.levelNum()))
    , dsoInitializer(nullptr)
    , isInitialized(false)
    , frameTracker(nullptr)
    , settings(_settings)
    , observers(observers) {
  LOG(INFO) << "create DsoSystem" << std::endl;

  marginalizedFrames.reserve(settings.expectedFramesCount);
  allFrames.reserve(settings.expectedFramesCount);

  for (int i = 0; i < cam->bundle.size(); ++i)
    pixelSelector[i] = PixelSelector(settings.pixelSelector);

  for (DsoObserver *obs : observers.dso)
    obs->created(this, cam, settings);
}

DsoSystem::~DsoSystem() {
  const KeyFrame *lastKeyFrames[Settings::max_maxKeyFrames];
  for (int i = 0; i < keyFrames.size(); ++i)
    lastKeyFrames[i] = keyFrames[i].get();
  for (DsoObserver *obs : observers.dso)
    obs->destructed(lastKeyFrames, keyFrames.size());
}

// returns reprojection + depth
std::pair<Vec2, double> reproject(CameraModel *cam, const SE3 origToReprojected,
                                  const Vec2 &p, double depth) {
  Vec3 reprojectedDir =
      origToReprojected * (depth * cam->unmap(p).normalized());
  Vec2 reprojectedPos = cam->map(reprojectedDir);
  return {reprojectedPos, reprojectedDir.norm()};
}

template <typename PointT> inline auto &getPoints(KeyFrameEntry &entry) {
  if constexpr (std::is_same_v<PointT, ImmaturePoint>)
    return entry.immaturePoints;
  else if constexpr (std::is_same_v<PointT, OptimizedPoint>)
    return entry.optimizedPoints;
  else
    CHECK(false) << "bad instantiation";
}

template <typename PointT> inline double getDepth(const PointT &p) {
  if constexpr (std::is_same_v<PointT, ImmaturePoint>)
    return p.depth;
  else if constexpr (std::is_same_v<PointT, OptimizedPoint>)
    return p.depth();
  else
    CHECK(false) << "bad instantiation";
}

template <typename PointT>
void DsoSystem::projectOntoBaseKf(Vec2 *points[],
                                  const std::optional<PointT ***> &refs,
                                  const std::optional<int **> &pointIndices,
                                  const std::optional<double **> &depths,
                                  int sizes[]) {
  for (int i = 0; i < cam->bundle.size(); ++i)
    sizes[i] = 0;

  for (const auto &kf : keyFrames) {
    SE3 refToBaseBody = baseFrame().thisToWorld.inverse() * kf->thisToWorld;
    for (int ri = 0; ri < cam->bundle.size(); ++ri) {
      const CameraBundle::CameraEntry &refCam = cam->bundle[ri];
      SE3 temp = refToBaseBody * refCam.thisToBody;
      for (int bi = 0; bi < cam->bundle.size(); ++bi) {
        CameraBundle::CameraEntry &baseCam = cam->bundle[bi];
        SE3 refToBase = baseCam.bodyToThis * temp;
        auto &pointVec = getPoints<PointT>(kf->frames[ri]);
        for (int pi = 0; pi < pointVec.size(); ++pi) {
          PointT &pnt = pointVec[pi];
          if (pnt.state != PointT::ACTIVE)
            continue;

          Vec3 baseDir = refToBase * (getDepth(pnt) * pnt.dir);
          Vec3 baseDirNormalized = baseDir.normalized();
          if (baseDirNormalized[2] < baseCam.cam.getMinZ())
            continue;
          Vec2 proj = baseCam.cam.map(baseDir);
          if (baseCam.cam.isOnImage(proj, 0)) {
            points[bi][sizes[bi]] = proj;
            if (refs)
              refs.value()[bi][sizes[bi]] = &pnt;
            if (pointIndices)
              pointIndices.value()[bi][sizes[bi]] = pi;
            if (depths)
              depths.value()[bi][sizes[bi]] = baseDir.norm();
            sizes[bi]++;
          }
        }
      }
    }
  }
}

class TimestampExtractor {
public:
  template <typename T> long long operator()(T pointer) {
    if constexpr (std::is_same_v<T, KeyFrame *>)
      return pointer->preKeyFrame->timestamp;
    else
      return pointer->timestamp;
  }
};

long long DsoSystem::getTimestamp(int frameNumber) {
  CHECK(frameNumber >= 0 && frameNumber < allFrames.size());
  return std::visit(TimestampExtractor(), allFrames[frameNumber]);
}

class FrameToWorldExtractor {
public:
  SE3 operator()(MarginalizedKeyFrame *f) { return f->thisToWorld; }
  SE3 operator()(MarginalizedPreKeyFrame *f) {
    return f->baseFrame->thisToWorld * f->baseToThis.inverse();
  }
  SE3 operator()(KeyFrame *f) { return f->thisToWorld; }
  SE3 operator()(PreKeyFrame *f) {
    return f->baseFrame->thisToWorld * f->baseToThis.inverse();
  }
};

SE3 DsoSystem::getFrameToWorld(int frameNumber) {
  CHECK(frameNumber >= 0 && frameNumber < allFrames.size());
  return std::visit(FrameToWorldExtractor(), allFrames[frameNumber]);
}

class LightWorldToFrameExtractor {
public:
  LightWorldToFrameExtractor(int ind)
      : ind(ind) {}

  AffLight operator()(MarginalizedKeyFrame *f) {
    return f->frames[ind].lightWorldToThis;
  }
  AffLight operator()(MarginalizedPreKeyFrame *f) {
    return f->lightBaseToFrame[ind] *
           f->baseFrame->frames[ind].lightWorldToThis;
  }
  AffLight operator()(KeyFrame *f) { return f->frames[ind].lightWorldToThis; }
  AffLight operator()(PreKeyFrame *f) {
    return f->frames[ind].lightBaseToThis *
           f->baseFrame->frames[ind].lightWorldToThis;
  }

private:
  int ind;
};

AffLight DsoSystem::getLightWorldToFrame(int frameNumber, int ind) {
  CHECK(frameNumber >= 0 && frameNumber < allFrames.size());
  return std::visit(LightWorldToFrameExtractor(ind), allFrames[frameNumber]);
}

double DsoSystem::getTimeLastByLbo() {
  CHECK(allFrames.size() >= 3);

  long long lboTime =
      getTimestamp(allFrames.size() - 2) - getTimestamp(allFrames.size() - 3);
  long long lastTime =
      getTimestamp(allFrames.size() - 1) - getTimestamp(allFrames.size() - 2);
  return double(lboTime) / lastTime;
}

SE3 predictScrewInternal(double timeLastByLbo, const SE3 &baseToLbo,
                         const SE3 &baseToLast) {
  SE3 lboToLast = baseToLast * baseToLbo.inverse();
  return SE3::exp(timeLastByLbo * lboToLast.log()) * baseToLast;
}

SE3 predictSimpleInternal(double timeLastByLbo, const SE3 &baseToLbo,
                          const SE3 &baseToLast) {
  SE3 lboToLast = baseToLast * baseToLbo.inverse();
  SO3 lastToCurRot = SO3::exp(timeLastByLbo * lboToLast.so3().log());
  Vec3 lastToCurTrans =
      timeLastByLbo *
      (lastToCurRot * lboToLast.so3().inverse() * lboToLast.translation());

  return SE3(lastToCurRot, lastToCurTrans) * baseToLast;
}

SE3 DsoSystem::predictInternal(double timeLastByLbo, const SE3 &baseToLbo,
                               const SE3 &baseToLast) {
  return settings.predictUsingScrew
             ? predictScrewInternal(timeLastByLbo, baseToLbo, baseToLast)
             : predictSimpleInternal(timeLastByLbo, baseToLbo, baseToLast);
}

SE3 DsoSystem::predictBaseKfToCur() {
  double timeLastByLbo = getTimeLastByLbo();

  SE3 baseToLbo =
      getFrameToWorld(allFrames.size() - 2).inverse() * baseFrame().thisToWorld;
  SE3 baseToLast =
      getFrameToWorld(allFrames.size() - 1).inverse() * baseFrame().thisToWorld;

  return predictInternal(timeLastByLbo, baseToLbo, baseToLast);
}

FrameTracker::TrackingResult DsoSystem::predictTracking() {
  FrameTracker::TrackingResult result;
  result.baseToTracked = predictBaseKfToCur();
  for (int i = 0; i < cam->bundle.size(); ++i)
    result.lightBaseToTracked[i] =
        getLightWorldToFrame(allFrames.size() - 1, i) *
        baseFrame().frames[i].lightWorldToThis.inverse();

  return result;
}

bool DsoSystem::doNeedKf(PreKeyFrame *lastFrame) {
  return allFrames.size() % settings.keyFrameDist() ==
         (settings.keyFrameDist() - 1);
}

void DsoSystem::addFrameTrackerObserver(FrameTrackerObserver *observer) {
  observers.frameTracker.push_back(observer);
}

void DsoSystem::marginalizeFrames() {
  if (keyFrames.size() > settings.maxKeyFrames()) {
    int count = static_cast<int>(keyFrames.size()) - settings.maxKeyFrames();
    const KeyFrame *marginalized[Settings::max_maxKeyFrames];
    for (int i = 0; i < count; ++i) {
      marginalized[i] = keyFrames[i].get();
      marginalizedFrames.emplace_back(new MarginalizedKeyFrame(*keyFrames[i]));
      allFrames[keyFrames[i]->preKeyFrame->globalFrameNum] =
          marginalizedFrames.back().get();
      for (int pi = 0; pi < keyFrames[i]->trackedFrames.size(); ++pi) {
        int num = keyFrames[i]->trackedFrames[pi]->globalFrameNum;
        allFrames[num] = marginalizedFrames.back()->trackedFrames[pi].get();
      }
    }

    for (DsoObserver *obs : observers.dso)
      obs->keyFramesMarginalized(marginalized, count);

    keyFrames.erase(keyFrames.begin(), keyFrames.begin() + count);
  }
}

void DsoSystem::activateNewOptimizedPoints() {
  StdVector<Vec2> optPoints[Settings::CameraBundle::max_camerasInBundle];
  Vec2 *optPointsPtrs[Settings::CameraBundle::max_camerasInBundle];
  int optSizes[Settings::CameraBundle::max_camerasInBundle];
  for (int i = 0; i < cam->bundle.size(); ++i) {
    optPoints[i].resize(settings.maxOptimizedPoints());
    optPointsPtrs[i] = optPoints[i].data();
    optSizes[i] = 0;
  }

  projectOntoBaseKf<OptimizedPoint>(optPointsPtrs, std::nullopt, std::nullopt,
                                    std::nullopt, optSizes);
  for (int i = 0; i < cam->bundle.size(); ++i)
    optPoints[i].resize(optSizes[i]);

  DistanceMap distMap(cam, optPoints, settings.distanceMap);

  StdVector<Vec2> projImm[Settings::CameraBundle::max_camerasInBundle];
  std::vector<ImmaturePoint *>
      refsImm[Settings::CameraBundle::max_camerasInBundle];
  std::vector<int> pointIndicesImm[Settings::CameraBundle::max_camerasInBundle];
  Vec2 *projImmPtrs[Settings::CameraBundle::max_camerasInBundle];
  ImmaturePoint **refsImmPtrs[Settings::CameraBundle::max_camerasInBundle];
  int *pointIndicesImmPtrs[Settings::CameraBundle::max_camerasInBundle];
  int immSizes[Settings::CameraBundle::max_camerasInBundle];
  int toResize =
      settings.maxKeyFrames() * settings.keyFrame.immaturePointsNum();
  for (int i = 0; i < cam->bundle.size(); ++i) {
    projImm[i].resize(toResize);
    refsImm[i].resize(toResize);
    pointIndicesImm[i].resize(toResize);
    projImmPtrs[i] = projImm[i].data();
    refsImmPtrs[i] = refsImm[i].data();
  }

  projectOntoBaseKf<ImmaturePoint>(projImmPtrs, std::make_optional(refsImmPtrs),
                                   std::make_optional(pointIndicesImmPtrs),
                                   std::nullopt, immSizes);
  std::vector<int> chosenInds[Settings::CameraBundle::max_camerasInBundle];
  for (int i = 0; i < cam->bundle.size(); ++i) {
    int areReady = 0;
    for (int j = 0; j < immSizes[i]; ++j)
      if (refsImm[i][j]->isReady())
        projImm[i][areReady++] = projImm[i][j];
    projImm[i].resize(areReady);
    chosenInds[i].reserve(areReady);
  }

  int curOptPoints = 0;
  for (const auto &kf : keyFrames)
    for (const auto &e : kf->frames)
      curOptPoints += e.optimizedPoints.size();

  int pointsNeeded = settings.maxOptimizedPoints() - curOptPoints;

  int chosenCount = distMap.choose(projImm, pointsNeeded, chosenInds);

  std::vector<std::pair<ImmaturePoint *, int>> chosenPoints;
  chosenPoints.reserve(chosenCount);
  for (int ci = 0; ci < cam->bundle.size(); ++ci)
    for (int j : chosenInds[ci])
      chosenPoints.push_back({refsImm[ci][j], pointIndicesImm[ci][j]});

  int oldSize = chosenPoints.size();
  std::sort(chosenPoints.begin(), chosenPoints.end(),
            [](const auto &p1, const auto &p2) {
              return p1.second == p2.second ? p1.first < p2.first
                                            : p1.second > p2.second;
            });
  chosenPoints.erase(std::unique(chosenPoints.begin(), chosenPoints.end()),
                     chosenPoints.end());

  LOG(INFO) << "\n\nNEW OPTIMIZED POINT SELECTION\n"
            << "Current # of OptimizedPoint-s = " << curOptPoints
            << ", needed = " << pointsNeeded << '\n'
            << ", duplicated = " << oldSize - chosenPoints.size() << '\n'
            << "New OptimizedPoint-s = " << chosenPoints.size() << std::endl;

  for (auto [p, ind] : chosenPoints) {
    KeyFrameEntry &entry = *p->host;
    entry.optimizedPoints.emplace_back(*p);
    std::swap(entry.immaturePoints[ind], entry.immaturePoints.back());
    entry.immaturePoints.pop_back();
  }
}

void DsoSystem::traceOn(const PreKeyFrame &frame) {}

void DsoSystem::addMultiFrame(const cv::Mat frames[], long long timestamp) {
  int globalFrameNum = allFrames.size();

  LOG(INFO) << "add frame #" << globalFrameNum << std::endl;

  if (!isInitialized) {
    LOG(INFO) << "put into initializer" << std::endl;
    isInitialized = dsoInitializer->addMultiFrame(frames);

    if (isInitialized) {
      LOG(INFO) << "initialization successful" << std::endl;
      DsoInitializer::InitializedVector init = dsoInitializer->initialize();
      for (const InitializedFrame &f : init)
        keyFrames.emplace_back(new KeyFrame(f, cam, globalFrameNum, timestamp, pixelSelector,
                               settings.keyFrame, settings.pyramid,
                               settings.getPointTracerSettings()));

      const KeyFrame *initializedKFs[Settings::max_maxKeyFrames];
      for (int i = 0; i < keyFrames.size(); ++i)
        initializedKFs[i] = keyFrames[i].get();

      for (DsoObserver *obs : observers.dso)
        obs->initialized(initializedKFs, keyFrames.size());

      // BundleAdjuster bundleAdjuster(cam, settings.bundleAdjuster,
      // settings.residualPattern, settings.gradWeighting, settings.intencity,
      // settings.affineLight, settings.threading, settings.depth);
      // for (auto &p : keyFrames)
      // bundleAdjuster.addKeyFrame(&p.second);
      // bundleAdjuster.adjust(settingMaxFirstBAIterations);

      int curPoints = 0;
      for (const auto &kf : keyFrames)
        for (const auto &e : kf->frames)
          curPoints += e.immaturePoints.size();
      StdVector<Vec2> points[Settings::CameraBundle::max_camerasInBundle];
      std::vector<double> depths[Settings::CameraBundle::max_camerasInBundle];
      int sizes[Settings::CameraBundle::max_camerasInBundle];
      Vec2 *pointPtrs[Settings::CameraBundle::max_camerasInBundle];
      double *depthPtrs[Settings::CameraBundle::max_camerasInBundle];
      for (int i = 0; i < cam->bundle.size(); ++i) {
        points[i].resize(curPoints);
        depths[i].resize(curPoints);
        pointPtrs[i] = points[i].data();
        depthPtrs[i] = depths[i].data();
      }

      projectOntoBaseKf<ImmaturePoint>(pointPtrs, std::nullopt, std::nullopt,
                                       std::make_optional(depthPtrs), sizes);
      FrameTracker::DepthedMultiFrame baseForTrack;
      for (int i = 0; i < cam->bundle.size(); ++i) {
        points[i].resize(sizes[i]);
        depths[i].resize(sizes[i]);
        std::vector<double> weights(points[i].size(), 1);
        baseForTrack.emplace_back(baseFrame().preKeyFrame->image(i),
                                  settings.pyramid.levelNum(), points[i].data(),
                                  depths[i].data(), weights.data(),
                                  points[i].size());
      }

      frameTracker = std::unique_ptr<FrameTracker>(
          new FrameTracker(camPyr.data(), baseForTrack, observers.frameTracker,
                           settings.getFrameTrackerSettings()));

      for (auto &kf : keyFrames)
        allFrames.push_back(kf.get());

      for (DsoObserver *obs : observers.dso)
        obs->newBaseFrame(baseFrame());
    }

    return;
  }

  std::unique_ptr<PreKeyFrame> preKeyFrame(new PreKeyFrame(
      &baseFrame(), cam, frames, globalFrameNum, timestamp, settings.pyramid));

  for (DsoObserver *obs : observers.dso)
    obs->newFrame(*preKeyFrame);

  FrameTracker::TrackingResult tracked =
      frameTracker->trackFrame(*preKeyFrame, predictTracking());

  preKeyFrame->baseToThis = tracked.baseToTracked;
  for (int i = 0; i < cam->bundle.size(); ++i)
    preKeyFrame->frames[i].lightBaseToThis = tracked.lightBaseToTracked[i];

  traceOn(*preKeyFrame);

  // for (DsoObserver *obs : observers.dso)
  // obs->pointsTraced ... ;

  bool needNewKf = doNeedKf(preKeyFrame.get());
  if (!needNewKf) {
    allFrames.push_back(preKeyFrame.get());
    baseFrame().trackedFrames.push_back(std::move(preKeyFrame));
  }

  if (settings.continueChoosingKeyFrames && needNewKf) {
    keyFrames.push_back(std::unique_ptr<KeyFrame>(
        new KeyFrame(std::move(preKeyFrame), pixelSelector, settings.keyFrame,
                     settings.getPointTracerSettings())));

    marginalizeFrames();
    activateNewOptimizedPoints();

    for (DsoObserver *obs : observers.dso)
      obs->newBaseFrame(baseFrame());

    if (settings.bundleAdjuster.runBA) {
      KeyFrame *kfPtrs[Settings::CameraBundle::max_camerasInBundle];
      for (int i = 0; i < keyFrames.size(); ++i)
        kfPtrs[i] = keyFrames[i].get();
      BundleAdjuster bundleAdjuster(cam, kfPtrs, keyFrames.size(),
                                    settings.getBundleAdjusterSettings());
      bundleAdjuster.adjust(settings.bundleAdjuster.maxIterations);
    }

    StdVector<Vec2> points[Settings::CameraBundle::max_camerasInBundle];
    std::vector<double> depths[Settings::CameraBundle::max_camerasInBundle];
    std::vector<OptimizedPoint *>
        refs[Settings::CameraBundle::max_camerasInBundle];
    Vec2 *pointsPtrs[Settings::CameraBundle::max_camerasInBundle];
    double *depthsPtrs[Settings::CameraBundle::max_camerasInBundle];
    OptimizedPoint **refsPtrs[Settings::CameraBundle::max_camerasInBundle];
    int sizes[Settings::CameraBundle::max_camerasInBundle];
    for (int i = 0; i < cam->bundle.size(); ++i) {
      points[i].resize(settings.maxOptimizedPoints());
      refs[i].resize(settings.maxOptimizedPoints());
      depths[i].resize(settings.maxOptimizedPoints());
      pointsPtrs[i] = points[i].data();
      refsPtrs[i] = refs[i].data();
      depthsPtrs[i] = depths[i].data();
    }

    projectOntoBaseKf<OptimizedPoint>(pointsPtrs, std::make_optional(refsPtrs),
                                      std::nullopt,
                                      std::make_optional(depthsPtrs), sizes);

    FrameTracker::DepthedMultiFrame baseForTrack;
    for (int i = 0; i < cam->bundle.size(); ++i) {
      points[i].resize(sizes[i]);
      refs[i].resize(sizes[i]);
      depths[i].resize(sizes[i]);
      std::vector<double> weights(points[i].size());
      for (int j = 0; j < points[i].size(); ++j)
        weights[j] = 1.0 / refs[i][j]->stddev;
      baseForTrack.emplace_back(
          baseFrame().preKeyFrame->image(i), settings.pyramid.levelNum(),
          points[i].data(), depths[i].data(), weights.data(), points[i].size());
    }

    frameTracker = std::unique_ptr<FrameTracker>(
        new FrameTracker(camPyr.data(), baseForTrack, observers.frameTracker,
                         settings.getFrameTrackerSettings()));
  }
}

} // namespace fishdso
