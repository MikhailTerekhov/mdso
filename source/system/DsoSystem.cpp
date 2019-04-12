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
          _settings.delaunayDsoInitializer, _settings.stereoMatcher,
          _settings.threading, _settings.triangulation, _settings.keyFrame,
          _settings.pointTracer, _settings.intencity, _settings.residualPattern,
          _settings.pyramid)))
    , isInitialized(false)
    , lastTrackRmse(INF)
    , firstFrameNum(-1)
    , settings(_settings)
    , observers(observers) {
  LOG(INFO) << "create DsoSystem" << std::endl;

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
depth<SetUniquePtr<ImmaturePoint>>(const SetUniquePtr<ImmaturePoint> &p) {
  return p->depth;
}

template <>
EIGEN_STRONG_INLINE double
depth<SetUniquePtr<OptimizedPoint>>(const SetUniquePtr<OptimizedPoint> &p) {
  return p->depth();
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

SE3 predictScrewInternal(int prevFramesSkipped, const SE3 &worldToBaseKf,
                         const SE3 &worldToLbo, const SE3 &worldToLast) {
  SE3 lboToLast = worldToLast * worldToLbo.inverse();
  double alpha = 1.0 / prevFramesSkipped;
  return SE3::exp(alpha * lboToLast.log()) * worldToLast *
         worldToBaseKf.inverse();
}

SE3 predictSimpleInternal(int prevFramesSkipped, const SE3 &worldToBaseKf,
                          const SE3 &worldToLbo, const SE3 &worldToLast) {
  SE3 lboToLast = worldToLast * worldToLbo.inverse();
  double alpha = 1.0 / prevFramesSkipped;
  SO3 lastToCurRot = SO3::exp(alpha * lboToLast.so3().log());
  Vec3 lastToCurTrans = alpha * (lastToCurRot * lboToLast.so3().inverse() *
                                 lboToLast.translation());

  return SE3(lastToCurRot, lastToCurTrans) * worldToLast *
         worldToBaseKf.inverse();
}

SE3 DsoSystem::predictInternal(int prevFramesSkipped, const SE3 &worldToBaseKf,
                               const SE3 &worldToLbo, const SE3 &worldToLast) {
  return settings.predictUsingScrew
             ? predictScrewInternal(prevFramesSkipped, worldToBaseKf,
                                    worldToLbo, worldToLast)
             : predictSimpleInternal(prevFramesSkipped, worldToBaseKf,
                                     worldToLbo, worldToLast);
}

SE3 DsoSystem::predictBaseKfToCur() {
  PreKeyFrame *baseKf = baseKeyFrame().preKeyFrame.get();

  int prevFramesSkipped =
      worldToFrame.rbegin()->first - (++worldToFrame.rbegin())->first;

  SE3 worldToLbo = (++worldToFrame.rbegin())->second;
  SE3 worldToLast = worldToFrame.rbegin()->second;

  return predictInternal(prevFramesSkipped, baseKf->worldToThis, worldToLbo,
                         worldToLast);
}

SE3 DsoSystem::purePredictBaseKfToCur() {
  PreKeyFrame *baseKf = baseKeyFrame().preKeyFrame.get();

  int prevFramesSkipped = worldToFramePredict.rbegin()->first -
                          (++worldToFramePredict.rbegin())->first;

  SE3 worldToLbo = (++worldToFramePredict.rbegin())->second;
  SE3 worldToLast = worldToFramePredict.rbegin()->second;

  return predictInternal(prevFramesSkipped, baseKf->worldToThis, worldToLbo,
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
  projectOntoBaseKf<OptimizedPoint>(&optPoints, nullptr, nullptr, nullptr);
  DistanceMap distMap(cam->getWidth(), cam->getHeight(), optPoints);

  StdVector<Vec2> immPoints;
  std::vector<ImmaturePoint *> immRefs;
  std::vector<KeyFrame *> kfs;
  projectOntoBaseKf<ImmaturePoint>(&immPoints, nullptr, &immRefs, &kfs);
  int oldIt = 0, newIt = 0;
  int totalActive = 0;
  for (; oldIt < immPoints.size(); ++oldIt) {
    if (immRefs[oldIt]->state == ImmaturePoint::ACTIVE)
      totalActive++;

    if (immRefs[oldIt]->isReady()) {
      immPoints[newIt] = immPoints[oldIt];
      kfs[newIt] = kfs[oldIt];
      immRefs[newIt++] = immRefs[oldIt];
    }
  }
  std::cout << "before selection:" << std::endl;
  std::cout << "total active = " << totalActive << std::endl;
  std::cout << "total good = " << newIt << std::endl;

  immPoints.resize(newIt);
  immRefs.resize(newIt);
  kfs.resize(newIt);
  int curOptPoints = std::accumulate(
      keyFrames.begin(), keyFrames.end(), 0, [](int acc, const auto &kfp) {
        return acc + kfp.second.optimizedPoints.size();
      });
  int pointsNeeded = settings.maxOptimizedPoints - curOptPoints;
  std::cout << "cur opt = " << curOptPoints << ", needed = " << pointsNeeded
            << std::endl;
  std::vector<int> activatedInd = distMap.choose(immPoints, pointsNeeded);
  std::cout << "chosen = " << activatedInd.size() << std::endl;
  for (int i : activatedInd) {
    kfs[i]->optimizedPoints.insert(
        SetUniquePtr<OptimizedPoint>(new OptimizedPoint(*immRefs[i])));
    int erasedCount = kfs[i]->immaturePoints.erase(makeFindPtr(immRefs[i]));
    CHECK(erasedCount > 0);
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

std::shared_ptr<PreKeyFrame> DsoSystem::addFrame(const cv::Mat &frame,
                                                 int globalFrameNum) {
  LOG(INFO) << "add frame #" << globalFrameNum << std::endl;

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

      if (settings.switchFirstMotionToGT || settings.allPosesGT) {
        SE3 worldToSecondKfGT =
            worldToFrameGT[keyFrames.rbegin()
                               ->second.preKeyFrame->globalFrameNum];
        keyFrames.rbegin()->second.preKeyFrame->worldToThis = worldToSecondKfGT;
        worldToFrame[keyFrames.rbegin()->second.preKeyFrame->globalFrameNum] =
            worldToSecondKfGT;
      }

      StdVector<Vec2> points;

      std::vector<double> depths;
      projectOntoBaseKf<ImmaturePoint>(&points, &depths, nullptr, nullptr);

      std::vector<double> weights(points.size(), 1.0);

      std::unique_ptr<DepthedImagePyramid> initialTrack(new DepthedImagePyramid(
          baseKeyFrame().preKeyFrame->frame(), settings.pyramid.levelNum,
          points, depths, weights));

      frameTracker = std::unique_ptr<FrameTracker>(new FrameTracker(
          camPyr, std::move(initialTrack), observers.frameTracker,
          settings.frameTracker, settings.pyramid, settings.affineLight,
          settings.intencity, settings.gradWeighting, settings.threading));

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

  if (settings.allPosesGT) {
    worldToFrame[globalFrameNum] = worldToFrameGT[globalFrameNum];
    preKeyFrame->worldToThis = worldToFrameGT[globalFrameNum];
  }

  lightKfToLast = lightBaseKfToCur;

  LOG(INFO) << "estimated motion\n"
            << baseKfToCur.translation() << "\n"
            << baseKfToCur.unit_quaternion().coeffs().transpose() << std::endl;

  SE3 diff = baseKfToCur * predicted.inverse();
  LOG(INFO) << "diff to predicted:\n"
            << diff.translation().norm() << "\n"
            << diff.so3().log().norm() << std::endl;
  LOG(INFO) << "estimated aff = \n" << lightBaseKfToCur << std::endl;

  if (settings.frameTracker.performTrackingCheckGT) {
    LOG(INFO) << "perform comparison to ground truth" << std::endl;
    checkLastTrackedGT(preKeyFrame.get());
  } else if (settings.frameTracker.performTrackingCheckStereo) {
    LOG(INFO) << "perform stereo check" << std::endl;
    checkLastTrackedStereo(preKeyFrame.get());
  }

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
                        settings.pointTracer, settings.intencity,
                        settings.residualPattern, settings.pyramid)));

    marginalizeFrames();
    activateNewOptimizedPoints();

    for (DsoObserver *obs : observers.dso)
      obs->newKeyFrame(&baseKeyFrame());

    if (settings.bundleAdjuster.runBA) {
      BundleAdjuster bundleAdjuster(
          cam, settings.bundleAdjuster, settings.residualPattern,
          settings.gradWeighting, settings.intencity, settings.affineLight,
          settings.threading, settings.depth);

      for (auto &kfp : keyFrames)
        bundleAdjuster.addKeyFrame(&kfp.second);
      bundleAdjuster.adjust(settings.bundleAdjuster.maxIterations);
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

    for (int i = 0; i < points.size(); ++i) {
      for (int pl = 0; pl < settings.pyramid.levelNum; ++pl) {
        cv::Point p = toCvPoint(points[i]);
        if (baseForTrack->depths[pl](p / (1 << pl)) <= 0) {
          std::cout << "pl=" << pl << " p=" << p << " psh=" << p / (1 << pl)
                    << " i=" << i << " d=" << depths[i] << " w=" << weights[i]
                    << " porig=" << points[i] << std::endl;
        }
      }
    }

    frameTracker = std::unique_ptr<FrameTracker>(new FrameTracker(
        camPyr, std::move(baseForTrack), observers.frameTracker,
        settings.frameTracker, settings.pyramid, settings.affineLight,
        settings.intencity, settings.gradWeighting, settings.threading));
  }

  return preKeyFrame;
}

void DsoSystem::checkLastTrackedGT(PreKeyFrame *lastFrame) {
  SE3 worldToBase = baseKeyFrame().preKeyFrame->worldToThis;
  SE3 worldToLast = lastFrame->worldToThis;
  SE3 baseToLast = worldToLast * worldToBase.inverse();

  SE3 worldToBaseGT =
      worldToFrameGT[baseKeyFrame().preKeyFrame->globalFrameNum];
  SE3 worldToLastGT = worldToFrameGT[lastFrame->globalFrameNum];
  SE3 baseToLastGT = worldToLastGT * worldToBaseGT.inverse();

  double transErr =
      (baseToLastGT.translation() - baseToLast.translation()).norm();
  double rotErr =
      180. / M_PI *
      (baseToLast.so3() * baseToLastGT.so3().inverse()).log().norm();

  LOG(INFO) << "translation error distance = " << transErr
            << "\nrotation error angle = " << rotErr << std::endl;
}

void DsoSystem::checkLastTrackedStereo(PreKeyFrame *lastFrame) {
  cv::Mat1b frames[2];
  frames[0] = keyFrames.begin()->second.preKeyFrame->frame();
  frames[1] = lastFrame->frame();

  SE3 worldToFirst = keyFrames.rbegin()->second.preKeyFrame->worldToThis;
  SE3 worldToBase = baseKeyFrame().preKeyFrame->worldToThis;

  StdVector<Vec2> points[2];
  std::vector<double> depths[2];
  StereoMatcher matcher(cam);
  SE3 matchedFirstToLast = matcher.match(frames, points, depths);
  SE3 refMotion = matchedFirstToLast * worldToFirst * worldToBase.inverse();
  SE3 trackedMotion = worldToFrame.rbegin()->second * worldToBase.inverse();

  double transErr = angle(refMotion.translation(), trackedMotion.translation());
  double rotErr =
      (trackedMotion.so3() * refMotion.so3().inverse()).log().norm();
  std::cout << "trans error angle = " << 180. / M_PI * transErr
            << "\nrot error angle = " << 180. / M_PI * rotErr << std::endl;
  std::cout << "rel rot error = " << rotErr / refMotion.so3().log().norm()
            << std::endl;

  refMotion.translation() *= trackedMotion.translation().norm();
  worldToFrameMatched[worldToFrame.rbegin()->first] = refMotion * worldToBase;
}

} // namespace fishdso
