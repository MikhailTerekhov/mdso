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
    , firstFrameNum(-1)
    , settings(_settings)
    , observers(observers) {
  LOG(INFO) << "create DsoSystem" << std::endl;

  frameNumbers.reserve(settings.initialMaxFrame);

  for (DsoObserver *obs : observers.dso)
    obs->created(this, cam, settings);
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
  for (auto &kfp : keyFrames) {
    auto &curPoints = getPoints<PointT>(kfp.second);
    if (&kfp.second == baseKf) {
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
      SE3 curToBase = baseKeyFrame().preKeyFrame->worldToThis *
                      kfp.second.preKeyFrame->worldToThis.inverse();
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
          kfs->push_back(&kfp.second);
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

SE3 predictScrewInternal(double timeLastByLbo, const SE3 &worldToBaseKf,
                         const SE3 &worldToLbo, const SE3 &worldToLast) {
  SE3 lboToLast = worldToLast * worldToLbo.inverse();
  return SE3::exp(timeLastByLbo * lboToLast.log()) * worldToLast *
         worldToBaseKf.inverse();
}

SE3 predictSimpleInternal(double timeLastByLbo, const SE3 &worldToBaseKf,
                          const SE3 &worldToLbo, const SE3 &worldToLast) {
  SE3 lboToLast = worldToLast * worldToLbo.inverse();
  SO3 lastToCurRot = SO3::exp(timeLastByLbo * lboToLast.so3().log());
  Vec3 lastToCurTrans =
      timeLastByLbo *
      (lastToCurRot * lboToLast.so3().inverse() * lboToLast.translation());

  return SE3(lastToCurRot, lastToCurTrans) * worldToLast *
         worldToBaseKf.inverse();
}

double DsoSystem::getTimeLastByLbo() {
  CHECK(frameNumbers.size() >= 3);

  int prevFramesSkipped = frameNumbers[frameNumbers.size() - 3] -
                          frameNumbers[frameNumbers.size() - 2];
  int lastFramesSkipped = frameNumbers[frameNumbers.size() - 2] -
                          frameNumbers[frameNumbers.size() - 1];
  return double(lastFramesSkipped) / prevFramesSkipped;
}

SE3 DsoSystem::predictInternal(double timeLastByLbo, const SE3 &worldToBaseKf,
                               const SE3 &worldToLbo, const SE3 &worldToLast) {
  return settings.predictUsingScrew
             ? predictScrewInternal(timeLastByLbo, worldToBaseKf, worldToLbo,
                                    worldToLast)
             : predictSimpleInternal(timeLastByLbo, worldToBaseKf, worldToLbo,
                                     worldToLast);
}

SE3 DsoSystem::predictBaseKfToCur() {
  PreKeyFrame *baseKf = baseKeyFrame().preKeyFrame.get();

  SE3 worldToLbo = worldToFrame[frameNumbers[frameNumbers.size() - 3]];
  SE3 worldToLast = worldToFrame[frameNumbers[frameNumbers.size() - 2]];

  return predictInternal(getTimeLastByLbo(), baseKf->worldToThis, worldToLbo,
                         worldToLast);
}

SE3 DsoSystem::purePredictBaseKfToCur() {
  PreKeyFrame *baseKf = baseKeyFrame().preKeyFrame.get();

  SE3 worldToLbo = worldToFramePredict[frameNumbers[frameNumbers.size() - 3]];
  SE3 worldToLast = worldToFramePredict[frameNumbers[frameNumbers.size() - 2]];

  return predictInternal(getTimeLastByLbo(), baseKf->worldToThis, worldToLbo,
                         worldToLast);
}

bool DsoSystem::doNeedKf(PreKeyFrame *lastFrame) {
  int shift = lastFrame->globalFrameNum - firstFrameNum;
  return shift > 0 && shift % 10 == 0;
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
    SE3 refToBase = baseKeyFrame().preKeyFrame->worldToThis *
                    kf.preKeyFrame->worldToThis.inverse();
    for (const auto &op : kf.optimizedPoints)
      if (op->state == OptimizedPoint::ACTIVE) {
        auto [p, depth] = (&kf == &baseKeyFrame()
                               ? std::pair(op->p, op->depth())
                               : reproject(cam, refToBase, op->p, op->depth()));
        optPoints.push_back(p);
      }
  }

  DistanceMap distMap(cam->getWidth(), cam->getHeight(), optPoints);

  int totalActive = 0;

  StdVector<Vec2> projectedImmatures;
  std::vector<std::pair<KeyFrame *, int>> immaturePositions;

  for (auto &[num, kf] : keyFrames) {
    SE3 refToBase = baseKeyFrame().preKeyFrame->worldToThis *
                    kf.preKeyFrame->worldToThis.inverse();
    for (int ind = 0; ind < kf.immaturePoints.size(); ++ind) {
      const auto &ip = kf.immaturePoints[ind];
      if (ip->state == ImmaturePoint::ACTIVE)
        totalActive++;
      if (ip->isReady()) {
        auto [p, depth] = (&kf == &baseKeyFrame()
                               ? std::pair(ip->p, ip->depth)
                               : reproject(cam, refToBase, ip->p, ip->depth));
        projectedImmatures.push_back(p);
        immaturePositions.push_back({&kf, ind});
      }
    }
  }

  std::cout << "before selection:" << std::endl;
  std::cout << "total active = " << totalActive << std::endl;
  std::cout << "total good = " << projectedImmatures.size() << std::endl;

  int curOptPoints = std::accumulate(
      keyFrames.begin(), keyFrames.end(), 0, [](int acc, const auto &kfp) {
        return acc + kfp.second.optimizedPoints.size();
      });
  int pointsNeeded = settings.maxOptimizedPoints - curOptPoints;
  std::cout << "cur opt = " << curOptPoints << ", needed = " << pointsNeeded
            << std::endl;

  std::vector<int> activatedIndices =
      distMap.choose(projectedImmatures, pointsNeeded);
  std::cout << "chosen = " << activatedIndices.size() << std::endl;
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
      std::vector<KeyFrame> kf = dsoInitializer->createKeyFrames();
      for (const auto &f : kf)
        worldToFramePredict[f.preKeyFrame->globalFrameNum] =
            worldToFrame[f.preKeyFrame->globalFrameNum] =
                f.preKeyFrame->worldToThis;
      for (KeyFrame &keyFrame : kf)
        keyFrames.insert(std::pair<int, KeyFrame>(
            keyFrame.preKeyFrame->globalFrameNum, std::move(keyFrame)));

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

      StdVector<DepthedImagePyramid::Point> pointsForPyr;
      for (const auto &[num, kf] : keyFrames) {
        SE3 refToBase = baseKeyFrame().preKeyFrame->worldToThis *
                        kf.preKeyFrame->worldToThis.inverse();
        for (const auto &ip : kf.immaturePoints)
          if (ip->state == ImmaturePoint::ACTIVE) {
            auto [p, depth] =
                (&kf == &baseKeyFrame()
                     ? std::pair(ip->p, ip->depth)
                     : reproject(cam, refToBase, ip->p, ip->depth));
            pointsForPyr.push_back({p, depth, 1.0});
          }
      }

      std::unique_ptr<DepthedImagePyramid> initialTrack(
          new DepthedImagePyramid(baseKeyFrame().preKeyFrame->frame(),
                                  settings.pyramid.levelNum, pointsForPyr));

      frameTracker = std::unique_ptr<FrameTracker>(new FrameTracker(
          camPyr, std::move(initialTrack), observers.frameTracker,
          settings.getFrameTrackerSettings()));

      lastInitialized = &keyFrames.rbegin()->second;

      firstFrameNum = lastInitialized->preKeyFrame->globalFrameNum;

      for (DsoObserver *obs : observers.dso)
        obs->newKeyFrame(&baseKeyFrame());

      return lastInitialized->preKeyFrame;
    }

    return nullptr;
  }

  std::shared_ptr<PreKeyFrame> preKeyFrame(
      new PreKeyFrame(cam, frame, globalFrameNum));

  for (DsoObserver *obs : observers.dso)
    obs->newFrame(preKeyFrame.get());

  SE3 baseKfToCur;
  AffineLightTransform<double> lightBaseKfToCur;

  SE3 purePredicted = purePredictBaseKfToCur();
  SE3 predicted = predictBaseKfToCur();

  LOG(INFO) << "start tracking this frame" << std::endl;

  std::tie(baseKfToCur, lightBaseKfToCur) = frameTracker->trackFrame(
      ImagePyramid(preKeyFrame->frame(), settings.pyramid.levelNum), predicted,
      lightKfToLast);

  LOG(INFO) << "tracking ended" << std::endl;

  PreKeyFrame *baseKf = baseKeyFrame().preKeyFrame.get();

  preKeyFrame->lightBaseToThis = lightBaseKfToCur;
  preKeyFrame->lightWorldToThis = lightBaseKfToCur * baseKf->lightWorldToThis;

  LOG(INFO) << "aff light: (fnum=" << preKeyFrame->globalFrameNum << ")\n"
            << preKeyFrame->lightWorldToThis << std::endl;

  worldToFrame[globalFrameNum] = baseKfToCur * baseKf->worldToThis;
  worldToFramePredict[globalFrameNum] = purePredicted * baseKf->worldToThis;

  preKeyFrame->baseToThis = baseKfToCur;
  preKeyFrame->worldToThis = worldToFrame[globalFrameNum];

  lightKfToLast = lightBaseKfToCur;

  LOG(INFO) << "estimated motion\n"
            << baseKfToCur.translation() << "\n"
            << baseKfToCur.unit_quaternion().coeffs().transpose() << std::endl;

  SE3 diff = baseKfToCur * predicted.inverse();
  LOG(INFO) << "diff to predicted:\n"
            << diff.translation().norm() << "\n"
            << diff.so3().log().norm() << std::endl;
  LOG(INFO) << "estimated aff = \n" << lightBaseKfToCur << std::endl;

  int totalActive = 0, totalGood = 0;
  constexpr int maxTraced = 5;
  std::vector<int> numTraced(maxTraced, 0);
  std::vector<int> numOnLevel(settings.pyramid.levelNum, 0);
  for (const auto &kfp : keyFrames)
    for (auto &ip : kfp.second.immaturePoints) {
      ip->traceOn(*preKeyFrame, ImmaturePoint::NO_DEBUG);
      if (!ip->lastTraced)
        continue;
      if (ip->numTraced < maxTraced)
        numTraced[ip->numTraced]++;
      if (ip->tracedPyrLevel >= 0)
        numOnLevel[ip->tracedPyrLevel]++;
      if (ip->state == ImmaturePoint::ACTIVE)
        totalActive++;
      if (ip->isReady())
        totalGood++;
    }
  std::cout << "active = " << totalActive << " good = " << totalGood
            << std::endl;
  std::cout << "succ traced = ";
  for (int i = 0; i < maxTraced; ++i)
    std::cout << numTraced[i] << ' ';
  std::cout << std::endl;
  std::cout << "traced on levels = ";
  for (int i = 0; i < settings.pyramid.levelNum; ++i)
    std::cout << numOnLevel[i] << ' ';
  std::cout << std::endl;

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

      for (auto &kfp : keyFrames)
        bundleAdjuster.addKeyFrame(&kfp.second);
      bundleAdjuster.adjust(settings.bundleAdjuster.maxIterations);
    }

    StdVector<DepthedImagePyramid::Point> pointsForPyr;
    for (const auto &[num, kf] : keyFrames) {
      SE3 refToBase = baseKeyFrame().preKeyFrame->worldToThis *
                      kf.preKeyFrame->worldToThis.inverse();
      for (const auto &op : kf.optimizedPoints)
        if (op->state == OptimizedPoint::ACTIVE) {
          auto [p, depth] =
              (&kf == &baseKeyFrame()
                   ? std::pair(op->p, op->depth())
                   : reproject(cam, refToBase, op->p, op->depth()));
          pointsForPyr.push_back({p, depth, 1.0 / op->stddev});
        }
    }

    std::unique_ptr<DepthedImagePyramid> baseForTrack(
        new DepthedImagePyramid(baseKeyFrame().preKeyFrame->frame(),
                                settings.pyramid.levelNum, pointsForPyr));

    frameTracker = std::unique_ptr<FrameTracker>(new FrameTracker(
        camPyr, std::move(baseForTrack), observers.frameTracker,
        settings.getFrameTrackerSettings()));
  }

  return preKeyFrame;
}

} // namespace fishdso
