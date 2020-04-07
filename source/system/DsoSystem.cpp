#include "system/DsoSystem.h"
#include "output/DsoObserver.h"
#include "output/FrameTrackerObserver.h"
#include "system/AffineLightTransform.h"
#include "system/DelaunayDsoInitializer.h"
#include "system/StereoMatcher.h"
#include "system/serialization.h"
#include "util/defs.h"
#include "util/geometry.h"
#include "util/settings.h"
#include <glog/logging.h>

namespace fishdso {

DsoSystem::DsoSystem(CameraModel *cam, const Observers &observers,
                     const Settings &_settings)
    : lastInitialized(nullptr)
    , scaleGTToOur(1.0)
    , cam(cam)
    , camPyr(cam->camPyr(_settings.pyramid.levelNum))
    , pixelSelector(_settings.pixelSelector)
    , dsoInitializer(std::unique_ptr<DsoInitializer>(new DelaunayDsoInitializer(
          this, cam, &pixelSelector, _settings.maxOptimizedPoints,
          DelaunayDsoInitializer::SPARSE_DEPTHS, observers.initializer,
          _settings.getInitializerSettings())))
    , isInitialized(false)
    , worldToFrame(_settings.initialMaxFrame)
    , worldToFramePredict(_settings.initialMaxFrame)
    , lastTrackRmse(INF)
    , settings(_settings)
    , observers(observers) {
  LOG(INFO) << "create DsoSystem" << std::endl;

  frameNumbers.reserve(settings.initialMaxFrame);

  for (DsoObserver *obs : observers.dso)
    obs->created(this, cam, settings);
}

DsoSystem::DsoSystem(const SnapshotLoader &snapshotLoader,
                     const Observers &observers, const Settings &_settings)
    : lastInitialized(nullptr)
    , scaleGTToOur(1.0)
    , cam(snapshotLoader.getCam())
    , camPyr(cam->camPyr(_settings.pyramid.levelNum))
    , pixelSelector(_settings.pixelSelector)
    , isInitialized(true)
    , worldToFrame(_settings.initialMaxFrame)
    , worldToFramePredict(_settings.initialMaxFrame)
    , lastTrackRmse(INF)
    , settings(_settings)
    , observers(observers) {
  LOG(INFO) << "create DsoSystem" << std::endl;

  frameNumbers.reserve(settings.initialMaxFrame);

  for (DsoObserver *obs : observers.dso)
    obs->created(this, cam, settings);

  snapshotLoader.load(keyFrames);
  CHECK_GE(keyFrames.size(), 2);

  for (auto &[keyFrameNum, keyFrame] : keyFrames) {
    frameNumbers.push_back(keyFrameNum);
    adjustWorldToFrameSizes(keyFrameNum);
    worldToFramePredict[keyFrameNum] = worldToFrame[keyFrameNum] =
        keyFrame.thisToWorld.inverse();
    for (DsoObserver *obs : observers.dso) {
      obs->newKeyFrame(&keyFrame);
      for (const auto &preKeyFrame : keyFrame.trackedFrames) {
        obs->newFrame(preKeyFrame.get());
        int preKeyFrameNum = preKeyFrame->globalFrameNum;
        frameNumbers.push_back(preKeyFrameNum);
        adjustWorldToFrameSizes(preKeyFrameNum);
        worldToFramePredict[preKeyFrameNum] = worldToFrame[preKeyFrameNum] =
            preKeyFrame->baseToThis * keyFrame.thisToWorld.inverse();
      }
    }
  }

  if (lastKeyFrame().trackedFrames.empty())
    lightKfToLast = lboKeyFrame().trackedFrames.back()->lightBaseToThis;
  else
    lightKfToLast = lastKeyFrame().trackedFrames.back()->lightBaseToThis;

  StdVector<Vec2> points;
  std::vector<double> depths;
  std::vector<double> weights(points.size());
  if (keyFrames.size() == 2) {
    std::vector<ImmaturePoint *> refs;
    projectOntoBaseKf<ImmaturePoint>(&points, &depths, &refs, nullptr);
    weights.resize(points.size());
    for (int i = 0; i < points.size(); ++i)
      weights[i] = 1.0 / refs[i]->stddev;
  } else {
    std::vector<OptimizedPoint *> refs;
    projectOntoBaseKf<OptimizedPoint>(&points, &depths, &refs, nullptr);
    weights.resize(points.size());
    for (int i = 0; i < points.size(); ++i)
      weights[i] = 1.0 / refs[i]->stddev;
  }

  std::unique_ptr<DepthedImagePyramid> baseForTrack(new DepthedImagePyramid(
      baseKeyFrame().preKeyFrame->frame(), settings.pyramid.levelNum, points,
      depths, weights));

  frameTracker = std::unique_ptr<FrameTracker>(
      new FrameTracker(camPyr, std::move(baseForTrack), observers.frameTracker,
                       settings.getFrameTrackerSettings()));
}

DsoSystem::~DsoSystem() {
  std::vector<const KeyFrame *> lastKeyFrames;
  lastKeyFrames.reserve(keyFrames.size());
  for (const auto &kfp : keyFrames)
    lastKeyFrames.push_back(&kfp.second);
  for (DsoObserver *obs : observers.dso)
    obs->destructed(lastKeyFrames);
}

template <typename PointT>
EIGEN_STRONG_INLINE auto &getPoints(const KeyFrame &keyFrame);

template <>
EIGEN_STRONG_INLINE auto &getPoints<ImmaturePoint>(const KeyFrame &keyFrame) {
  return keyFrame.immaturePoints;
}

template <>
EIGEN_STRONG_INLINE auto &getPoints<OptimizedPoint>(const KeyFrame &keyFrame) {
  return keyFrame.optimizedPoints;
}

template <typename PtrPointT>
EIGEN_STRONG_INLINE double depth(const PtrPointT &p);

template <>
EIGEN_STRONG_INLINE double
depth<std::unique_ptr<ImmaturePoint>>(const std::unique_ptr<ImmaturePoint> &p) {
  return p->depth;
}

template <>
EIGEN_STRONG_INLINE double depth<std::unique_ptr<OptimizedPoint>>(
    const std::unique_ptr<OptimizedPoint> &p) {
  return p->depth();
}

// returns reprojection + depth
std::pair<Vec2, double> reproject(CameraModel *cam, const SE3 origToReprojected,
                                  const Vec2 &p, double depth) {
  Vec3 reprojectedDir =
      origToReprojected * (depth * cam->unmap(p).normalized());
  Vec2 reprojectedPos = cam->map(reprojectedDir);
  return {reprojectedPos, reprojectedDir.norm()};
}

template <typename PointT>
void DsoSystem::projectOntoBaseKf(StdVector<Vec2> *points,
                                  std::vector<double> *depths,
                                  std::vector<PointT *> *ptrs,
                                  std::vector<KeyFrame *> *kfs) {
  if (points)
    points->resize(0);
  if (depths)
    depths->resize(0);
  if (ptrs)
    ptrs->resize(0);
  if (kfs)
    kfs->resize(0);

  KeyFrame *baseKf = &baseKeyFrame();
  for (auto &[num, kf] : keyFrames) {
    auto &curPoints = getPoints<PointT>(kf);
    if (&kf == baseKf) {
      for (auto &p : curPoints)
        if (p->state == PointT::ACTIVE) {
          if (points)
            points->push_back(p->p);
          if (depths)
            depths->push_back(depth(p));
          if (ptrs)
            ptrs->push_back(p.get());
          if (kfs)
            kfs->push_back(baseKf);
        }
    } else {
      SE3 curToBase = baseKf->thisToWorld.inverse() * kf.thisToWorld;
      for (auto &p : curPoints) {
        Vec3 baseDir = curToBase * (depth(p) * cam->unmap(p->p).normalized());
        Vec2 basePos = cam->map(baseDir);
        if (!cam->isOnImage(basePos, 0))
          continue;
        if (points)
          points->push_back(basePos);
        if (depths)
          depths->push_back(baseDir.norm());
        if (ptrs)
          ptrs->push_back(p.get());
        if (kfs)
          kfs->push_back(&kf);
      }
    }
  }
}

template void DsoSystem::projectOntoBaseKf<ImmaturePoint>(
    StdVector<Vec2> *points, std::vector<double> *depths,
    std::vector<ImmaturePoint *> *ptrs, std::vector<KeyFrame *> *kfs);
template void DsoSystem::projectOntoBaseKf<OptimizedPoint>(
    StdVector<Vec2> *points, std::vector<double> *depths,
    std::vector<OptimizedPoint *> *ptrs, std::vector<KeyFrame *> *kfs);

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

double DsoSystem::getTimeLastByLbo() {
  CHECK(frameNumbers.size() >= 3);

  int prevFramesSkipped = frameNumbers[frameNumbers.size() - 3] -
                          frameNumbers[frameNumbers.size() - 2];
  int lastFramesSkipped = frameNumbers[frameNumbers.size() - 2] -
                          frameNumbers[frameNumbers.size() - 1];
  return double(lastFramesSkipped) / prevFramesSkipped;
}

SE3 DsoSystem::predictInternal(double timeLastByLbo, const SE3 &baseToLbo,
                               const SE3 &baseToLast) {
  return settings.predictUsingScrew
             ? predictScrewInternal(timeLastByLbo, baseToLbo, baseToLast)
             : predictSimpleInternal(timeLastByLbo, baseToLbo, baseToLast);
}

SE3 DsoSystem::predictBaseKfToCur() {
  double timeLastByLbo = getTimeLastByLbo();

  SE3 baseToLbo = worldToFrame[frameNumbers[frameNumbers.size() - 3]] *
                  baseKeyFrame().thisToWorld;
  SE3 baseToLast = worldToFrame[frameNumbers[frameNumbers.size() - 2]] *
                   baseKeyFrame().thisToWorld;

  return predictInternal(timeLastByLbo, baseToLbo, baseToLast);
}

SE3 DsoSystem::purePredictBaseKfToCur() {
  SE3 baseToLbo = worldToFramePredict[frameNumbers[frameNumbers.size() - 3]] *
                  baseKeyFrame().thisToWorld;
  SE3 baseToLast = worldToFramePredict[frameNumbers[frameNumbers.size() - 2]] *
                   baseKeyFrame().thisToWorld;

  return predictInternal(getTimeLastByLbo(), baseToLbo, baseToLast);
}

bool DsoSystem::doNeedKf(PreKeyFrame *lastFrame) {
  int shift =
      lastFrame->globalFrameNum - lastKeyFrame().preKeyFrame->globalFrameNum;
  return shift > 0 && shift % settings.shiftBetweenKeyFrames == 0;
}

void DsoSystem::addFrameTrackerObserver(FrameTrackerObserver *observer) {
  observers.frameTracker.push_back(observer);
  if (frameTracker)
    frameTracker->addObserver(observer);
}

void DsoSystem::marginalizeFrames() {
  if (keyFrames.size() > settings.maxKeyFrames) {
    int count = static_cast<int>(keyFrames.size()) - settings.maxKeyFrames;
    std::vector<const KeyFrame *> marginalized;
    marginalized.reserve(count);
    auto kfIt = keyFrames.begin();
    for (int i = 0; i < count; ++i, ++kfIt)
      marginalized.push_back(&kfIt->second);
    for (DsoObserver *obs : observers.dso)
      obs->keyFramesMarginalized(marginalized);

    for (int i = 0; i < count; ++i)
      keyFrames.erase(keyFrames.begin());
  }
}

void DsoSystem::activateNewOptimizedPoints() {
  StdVector<Vec2> optPoints;
  for (const auto &[num, kf] : keyFrames) {
    SE3 refToBase = baseKeyFrame().thisToWorld.inverse() * kf.thisToWorld;
    for (const auto &op : kf.optimizedPoints)
      if (op->state == OptimizedPoint::ACTIVE) {
        auto [p, depth] = (&kf == &baseKeyFrame()
                               ? std::pair(op->p, op->depth())
                               : reproject(cam, refToBase, op->p, op->depth()));
        optPoints.push_back(p);
      }
  }

  DistanceMap distMap(cam->getWidth(), cam->getHeight(), optPoints);

  StdVector<Vec2> projectedImmatures;
  std::vector<std::pair<KeyFrame *, int>> immaturePositions;

  for (auto &[num, kf] : keyFrames) {
    SE3 refToBase = baseKeyFrame().thisToWorld.inverse() * kf.thisToWorld;
    for (int ind = 0; ind < kf.immaturePoints.size(); ++ind) {
      const auto &ip = kf.immaturePoints[ind];
      if (ip->isReady()) {
        auto [p, depth] = (&kf == &baseKeyFrame()
                               ? std::pair(ip->p, ip->depth)
                               : reproject(cam, refToBase, ip->p, ip->depth));
        projectedImmatures.push_back(p);
        immaturePositions.push_back({&kf, ind});
      }
    }
  }

  LOG(INFO) << "\n\nPOINT SELECTION\n"
            << "Ready to be optimized = " << projectedImmatures.size()
            << std::endl;

  int curOptPoints = std::accumulate(
      keyFrames.begin(), keyFrames.end(), 0, [](int acc, const auto &kfp) {
        return acc + kfp.second.optimizedPoints.size();
      });
  int pointsNeeded = settings.maxOptimizedPoints - curOptPoints;
  LOG(INFO) << "Current # of OptimizedPoint-s = " << curOptPoints
            << ", needed = " << pointsNeeded << '\n';

  std::vector<int> activatedIndices =
      distMap.choose(projectedImmatures, pointsNeeded);
  LOG(INFO) << "New OptimizedPoint-s = " << activatedIndices.size()
            << std::endl;
  std::sort(activatedIndices.begin(), activatedIndices.end(),
            [&immaturePositions](int i1, int i2) {
              return immaturePositions[i1].second >
                     immaturePositions[i2].second;
            });

  for (int i : activatedIndices) {
    KeyFrame *kf = immaturePositions[i].first;
    auto &ip = kf->immaturePoints[immaturePositions[i].second];
    kf->optimizedPoints.push_back(
        std::unique_ptr<OptimizedPoint>(new OptimizedPoint(*ip)));

    std::swap(ip, kf->immaturePoints.back());
    kf->immaturePoints.pop_back();
  }
}

bool DsoSystem::didTrackFail() {
  return frameTracker->lastRmse >
         lastTrackRmse * settings.frameTracker.trackFailFactor;
}

std::pair<SE3, AffineLightTransform<double>>
DsoSystem::recoverTrack(PreKeyFrame *lastFrame) {
  // TODO
  typedef std::pair<SE3, AffineLightTransform<double>> retT;
  return retT();
}

void DsoSystem::adjustWorldToFrameSizes(int newFrameNum) {
  CHECK(newFrameNum > 0);
  if (newFrameNum >= worldToFrame.size()) {
    worldToFrame.resize(newFrameNum + 1);
    worldToFramePredict.resize(newFrameNum + 1);
  }
}

std::shared_ptr<PreKeyFrame> DsoSystem::addFrame(const cv::Mat &frame,
                                                 int globalFrameNum) {
  LOG(INFO) << "add frame #" << globalFrameNum << std::endl;

  adjustWorldToFrameSizes(globalFrameNum);
  CHECK(frameNumbers.empty() || globalFrameNum > frameNumbers.back());
  frameNumbers.push_back(globalFrameNum);

  if (!isInitialized) {
    LOG(INFO) << "put into initializer" << std::endl;
    isInitialized = dsoInitializer->addFrame(frame, globalFrameNum);

    if (isInitialized) {
      LOG(INFO) << "initialization successful" << std::endl;
      StdVector<KeyFrame> kf = dsoInitializer->createKeyFrames();
      for (const auto &f : kf)
        worldToFramePredict[f.preKeyFrame->globalFrameNum] =
            worldToFrame[f.preKeyFrame->globalFrameNum] =
                f.thisToWorld.inverse();
      for (KeyFrame &keyFrame : kf) {
        int num = keyFrame.preKeyFrame->globalFrameNum;
        keyFrames.insert(std::pair<int, KeyFrame>(num, std::move(keyFrame)));
      }

      std::vector<const KeyFrame *> initializedKFs;
      initializedKFs.reserve(keyFrames.size());
      for (const auto &kfp : keyFrames)
        initializedKFs.push_back(&kfp.second);

      for (DsoObserver *obs : observers.dso)
        obs->initialized(initializedKFs);

      // BundleAdjuster bundleAdjuster(cam, settings.bundleAdjuster,
      // settings.residualPattern, settings.gradWeighting, settings.intencity,
      // settings.affineLight, settings.threading, settings.depth);
      // for (auto &p : keyFrames)
      // bundleAdjuster.addKeyFrame(&p.second);
      // bundleAdjuster.adjust(settingMaxFirstBAIterations);

      StdVector<Vec2> points;

      std::vector<double> depths;
      projectOntoBaseKf<ImmaturePoint>(&points, &depths, nullptr, nullptr);

      std::vector<double> weights(points.size(), 1.0);

      std::unique_ptr<DepthedImagePyramid> initialTrack(new DepthedImagePyramid(
          baseKeyFrame().preKeyFrame->frame(), settings.pyramid.levelNum,
          points, depths, weights));

      frameTracker = std::unique_ptr<FrameTracker>(new FrameTracker(
          camPyr, std::move(initialTrack), observers.frameTracker,
          settings.getFrameTrackerSettings()));

      lastInitialized = &keyFrames.rbegin()->second;

      for (DsoObserver *obs : observers.dso)
        obs->newKeyFrame(&baseKeyFrame());

      return lastInitialized->preKeyFrame;
    }

    return nullptr;
  }

  std::shared_ptr<PreKeyFrame> preKeyFrame(
      new PreKeyFrame(&baseKeyFrame(), cam, frame, globalFrameNum));

  for (DsoObserver *obs : observers.dso)
    obs->newFrame(preKeyFrame.get());

  SE3 baseKfToCur;
  AffineLightTransform<double> lightBaseKfToCur;

  SE3 purePredicted = purePredictBaseKfToCur();
  SE3 predicted = predictBaseKfToCur();

  std::tie(baseKfToCur, lightBaseKfToCur) =
      frameTracker->trackFrame(*preKeyFrame, predicted, lightKfToLast);

  preKeyFrame->lightBaseToThis = lightBaseKfToCur;

  LOG(INFO) << "aff light (base to cur): (fnum=" << preKeyFrame->globalFrameNum
            << ")\n"
            << preKeyFrame->lightBaseToThis << std::endl;

  worldToFrame[globalFrameNum] =
      baseKfToCur * baseKeyFrame().thisToWorld.inverse();
  worldToFramePredict[globalFrameNum] =
      purePredicted * baseKeyFrame().thisToWorld.inverse();

  preKeyFrame->baseToThis = baseKfToCur;

  lightKfToLast = lightBaseKfToCur;

  SE3 diff = baseKfToCur * predicted.inverse();
  LOG(INFO) << "diff to predicted (trans and rot): "
            << diff.translation().norm() << " " << diff.so3().log().norm()
            << '\n';

  int totalTraced = 0, totalGood = 0;
  constexpr int maxTraced = 8;
  std::vector<int> numTraced(maxTraced, 0);
  std::vector<int> numOnLevel(settings.pyramid.levelNum, 0);
  for (const auto &[num, kf] : keyFrames)
    for (auto &ip : kf.immaturePoints) {
      auto status = ip->traceOn(kf, *preKeyFrame, ImmaturePoint::NO_DEBUG);

      if (status == ImmaturePoint::OK)
        totalTraced++;
      if (ip->isReady())
        totalGood++;

      if (ip->numTraced < maxTraced)
        numTraced[ip->numTraced]++;
      if (ip->tracedPyrLevel >= 0)
        numOnLevel[ip->tracedPyrLevel]++;
    }

  LOG(INFO) << "POINT TRACING:";
  LOG(INFO) << "Successfully traced = " << totalTraced << "\n";
  LOG(INFO) << "Ready to be optimized = " << totalGood << "\n";
  LOG(INFO) << "Traced by number: ";
  outputArrayUndivided(LOG(INFO), numTraced.data(), maxTraced);
  LOG(INFO) << "Last traced on pyramid levels: ";
  outputArrayUndivided(LOG(INFO), numOnLevel.data(), settings.pyramid.levelNum);

  // for (DsoObserver *obs : observers.dso)
  // obs->pointsTraced ... ;

  bool needNewKf = doNeedKf(preKeyFrame.get());
  if (!needNewKf)
    baseKeyFrame().trackedFrames.push_back(preKeyFrame);

  if (settings.continueChoosingKeyFrames && needNewKf) {
    int kfNum = preKeyFrame->globalFrameNum;
    keyFrames.insert(std::pair<int, KeyFrame>(
        kfNum, KeyFrame(preKeyFrame, pixelSelector, settings.keyFrame,
                        settings.getPointTracerSettings())));

    marginalizeFrames();
    activateNewOptimizedPoints();

    for (DsoObserver *obs : observers.dso)
      obs->newKeyFrame(&baseKeyFrame());

    if (settings.bundleAdjuster.runBA) {
      BundleAdjuster bundleAdjuster(cam, settings.getBundleAdjusterSettings());

      for (auto &[num, kf] : keyFrames)
        bundleAdjuster.addKeyFrame(&kf);
      bundleAdjuster.adjust(settings.bundleAdjuster.maxIterations);

      for (const auto &[num, kf] : keyFrames) {
        SE3 worldToKf = kf.thisToWorld.inverse();
        worldToFrame[kf.preKeyFrame->globalFrameNum] = worldToKf;
        for (const auto &pkf : kf.trackedFrames)
          worldToFrame[pkf->globalFrameNum] = pkf->baseToThis * worldToKf;
      }
    }

    StdVector<Vec2> points;
    std::vector<double> depths;
    std::vector<OptimizedPoint *> refs;
    projectOntoBaseKf<OptimizedPoint>(&points, &depths, &refs, nullptr);
    std::vector<double> weights(points.size());
    for (int i = 0; i < points.size(); ++i)
      weights[i] = 1.0 / refs[i]->stddev;
    std::unique_ptr<DepthedImagePyramid> baseForTrack(new DepthedImagePyramid(
        baseKeyFrame().preKeyFrame->frame(), settings.pyramid.levelNum, points,
        depths, weights));

    // for (int i = 0; i < points.size(); ++i) {
    // for (int pl = 0; pl < settings.pyramid.levelNum; ++pl) {
    // cv::Point p = toCvPoint(points[i]);
    // if (baseForTrack->depths[pl](p / (1 << pl)) <= 0) {
    // std::cout << "pl=" << pl << " p=" << p << " psh=" << p / (1 << pl)
    // << " i=" << i << " d=" << depths[i] << " w=" << weights[i]
    // << " porig=" << points[i] << std::endl;
    // }
    // }
    // }

    frameTracker = std::unique_ptr<FrameTracker>(new FrameTracker(
        camPyr, std::move(baseForTrack), observers.frameTracker,
        settings.getFrameTrackerSettings()));
  }

  return preKeyFrame;
}

void DsoSystem::saveSnapshot(const std::string &snapshotDir) const {
  SnapshotSaver snapshotSaver(snapshotDir,
                              settings.residualPattern.pattern().size());
  std::vector<const KeyFrame *> keyFramePtrs;
  keyFramePtrs.reserve(keyFrames.size());
  for (const auto &[frameNum, keyFrame] : keyFrames)
    keyFramePtrs.push_back(&keyFrame);
  snapshotSaver.save(keyFramePtrs.data(), keyFramePtrs.size());
}

} // namespace fishdso
