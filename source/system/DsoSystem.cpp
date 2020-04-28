#include "system/DsoSystem.h"
#include "output/DsoObserver.h"
#include "output/FrameTrackerObserver.h"
#include "system/BundleAdjusterCeres.h"
#include "system/BundleAdjusterSelfMade.h"
#include "system/DelaunayDsoInitializer.h"
#include "system/Reprojector.h"
#include "system/TrackingPredictorRot.h"
#include "system/TrackingPredictorScrew.h"
#include "system/serialization.h"
#include "util/flags.h"
#include "util/settings.h"
#include <glog/logging.h>
#include <optional>

#define PH (settings.residualPattern.height)

namespace mdso {

DsoSystem::DsoSystem(CameraBundle *cam, Preprocessor *preprocessor,
                     const Observers &observers, const Settings &_settings)
    : cam(cam)
    , camPyr(cam->camPyr(_settings.pyramid.levelNum()))
    , isInitialized(false)
    , frameTracker(nullptr)
    , settings(_settings)
    , pointTracerSettings(_settings.getPointTracerSettings())
    , observers(observers)
    , preprocessor(preprocessor)
    , trackingPredictor(settings.predictUsingScrew
                            ? std::unique_ptr<TrackingPredictor>(
                                  new TrackingPredictorScrew(this))
                            : std::unique_ptr<TrackingPredictor>(
                                  new TrackingPredictorRot(this)))
    , wasRestored(false) {
  LOG(INFO) << "create DsoSystem" << std::endl;

  marginalizedFrames.reserve(settings.expectedFramesCount);
  keyFrames.reserve(settings.maxKeyFrames() + 1);
  allFrames.reserve(settings.expectedFramesCount);

  pixelSelector.reserve(cam->bundle.size());
  for (int i = 0; i < cam->bundle.size(); ++i)
    pixelSelector.emplace_back(settings.pixelSelector);

  dsoInitializer.reset(new DelaunayDsoInitializer(
      this, cam, pixelSelector.data(), observers.initializer,
      settings.getInitializerSettings()));

  for (DsoObserver *obs : observers.dso)
    obs->created(this, cam, settings);
}

DsoSystem::DsoSystem(std::vector<std::unique_ptr<KeyFrame>> &restoredKeyFrames,
                     CameraBundle *_cam, Preprocessor *_preprocessor,
                     const Observers &_observers, const Settings &_settings)
    : cam(_cam)
    , camPyr(cam->camPyr(_settings.pyramid.levelNum()))
    , isInitialized(true)
    , settings(_settings)
    , pointTracerSettings(_settings.getPointTracerSettings())
    , observers(_observers)
    , preprocessor(_preprocessor)
    , trackingPredictor(settings.predictUsingScrew
                            ? std::unique_ptr<TrackingPredictor>(
                                  new TrackingPredictorScrew(this))
                            : std::unique_ptr<TrackingPredictor>(
                                  new TrackingPredictorRot(this)))
    , wasRestored(true) {
  LOG(INFO) << "create DsoSystem from a snapshot" << std::endl;
  CHECK(settings.trackFromLastKf) << "restoration from a snapshot is supported "
                                     "when using the last base frame only.";

  marginalizedFrames.reserve(settings.expectedFramesCount);
  keyFrames.reserve(settings.maxKeyFrames() + 1);

  for (DsoObserver *obs : observers.dso)
    obs->created(this, cam, settings);

  for (auto &keyFrame : restoredKeyFrames) {
    keyFrames.push_back(std::move(keyFrame));
    for (DsoObserver *obs : observers.dso) {
      obs->newBaseFrame(*keyFrames.back());
      for (const auto &preKeyFrame : keyFrames.back()->trackedFrames)
        obs->newFrame(*preKeyFrame);
    }
  }

  initializeAllFrames();

  pixelSelector.reserve(cam->bundle.size());
  for (int i = 0; i < cam->bundle.size(); ++i)
    pixelSelector.emplace_back(settings.pixelSelector);

  // If there are only 2 keyframes, we decide that the system was only recently
  // initialized, thus tracking was done over immature points.
  FrameTracker::DepthedMultiFrame baseForTrack =
      keyFrames.size() == 2 ? getBaseForTrack<ImmaturePoint>()
                            : getBaseForTrack<OptimizedPoint>();
  frameTracker = std::unique_ptr<FrameTracker>(new FrameTracker(
      camPyr.data(), baseForTrack, baseFrame(), observers.frameTracker,
      settings.getFrameTrackerSettings()));
}

DsoSystem::~DsoSystem() {
  std::vector<const KeyFrame *> lastKeyFrames(keyFrames.size());
  for (int i = 0; i < keyFrames.size(); ++i)
    lastKeyFrames[i] = keyFrames[i].get();
  for (DsoObserver *obs : observers.dso)
    obs->destructed(lastKeyFrames.data(), keyFrames.size());
}

std::vector<const KeyFrame *> DsoSystem::getKeyFrames() const {
  std::vector<const KeyFrame *> result;
  result.reserve(keyFrames.size());
  for (const auto &kf : keyFrames)
    result.push_back(kf.get());
  return result;
}

int DsoSystem::trajectorySize() const {
  if (allFrames.size() > 0 &&
      std::holds_alternative<PreKeyFrame *>(allFrames.back())) {
    PreKeyFrame *last = std::get<PreKeyFrame *>(allFrames.back());
    return last->wasTracked() ? allFrames.size() : int(allFrames.size()) - 1;
  }
  return allFrames.size();
}

int DsoSystem::camNumber() const { return cam->bundle.size(); }

class TimestampExtractor {
public:
  TimestampExtractor(int numCams)
      : numCams(numCams) {}

  template <typename T> Timestamp operator()(T pointer) {
    if constexpr (std::is_same_v<T, KeyFrame *>) {
      auto &frames = pointer->preKeyFrame->frames;
      Timestamp sum = 0;
      for (int i = 0; i < numCams; ++i)
        sum += frames[i].timestamp;
      return sum / numCams;
    } else {
      auto &frames = pointer->frames;
      Timestamp sum = 0;
      for (int i = 0; i < numCams; ++i)
        sum += frames[i].timestamp;
      return sum / numCams;
    }
  }

private:
  int numCams;
};

Timestamp DsoSystem::timestamp(int ind) const {
  CHECK(ind >= 0 && ind < allFrames.size());
  return std::visit(TimestampExtractor(cam->bundle.size()), allFrames[ind]);
}

class FrameToWorldExtractor {
public:
  SE3 operator()(const MarginalizedKeyFrame *f) { return f->thisToWorld; }
  SE3 operator()(const MarginalizedPreKeyFrame *f) {
    return f->baseFrame->thisToWorld * f->baseToThis.inverse();
  }
  SE3 operator()(const KeyFrame *f) { return f->thisToWorld(); }
  SE3 operator()(const PreKeyFrame *f) {
    return f->baseFrame->thisToWorld() * f->baseToThis().inverse();
  }
};

SE3 DsoSystem::bodyToWorld(int ind) const {
  CHECK(ind >= 0 && ind < trajectorySize());
  return std::visit(FrameToWorldExtractor(), allFrames[ind]);
}

class LightWorldToFrameExtractor {
public:
  LightWorldToFrameExtractor(int ind)
      : ind(ind) {}

  AffLight operator()(MarginalizedKeyFrame *f) {
    return f->frames[ind].lightWorldToThis;
  }
  AffLight operator()(MarginalizedPreKeyFrame *f) {
    return f->frames[ind].lightBaseToThis *
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

AffLight DsoSystem::affLightWorldToBody(int ind, int camInd) const {
  CHECK(ind >= 0 && ind < allFrames.size());
  return std::visit(LightWorldToFrameExtractor(camInd), allFrames[ind]);
}

class GlobalFrameNumSetter {
public:
  GlobalFrameNumSetter(int globalFrameNum)
      : globalFrameNum(globalFrameNum) {}

  void operator()(KeyFrame *keyFrame) {
    keyFrame->preKeyFrame->globalFrameNum = globalFrameNum;
  }
  void operator()(PreKeyFrame *preKeyFrame) {
    preKeyFrame->globalFrameNum = globalFrameNum;
  }
  void operator()(MarginalizedKeyFrame *) { CHECK(false); }
  void operator()(MarginalizedPreKeyFrame *) { CHECK(false); }

private:
  int globalFrameNum;
};

void DsoSystem::initializeAllFrames() {
  allFrames.reserve(settings.expectedFramesCount);

  std::vector<std::pair<Timestamp, FramePointer>> frames;
  TimestampExtractor timestampExtractor(cam->bundle.size());
  for (const auto &keyFrame : keyFrames) {
    frames.push_back(
        {timestampExtractor(keyFrame.get()), FramePointer(keyFrame.get())});
    for (const auto &preKeyFrame : keyFrame->trackedFrames)
      frames.push_back({timestampExtractor(preKeyFrame.get()),
                        FramePointer(preKeyFrame.get())});
  }
  std::sort(frames.begin(), frames.end(),
            [](auto f1, auto f2) { return f1.first < f2.first; });
  for (auto [ts, frame] : frames) {
    std::visit(GlobalFrameNumSetter(allFrames.size()), frame);
    allFrames.push_back(frame);
  }
}

int DsoSystem::totalOptimized() const {
  int curOptPoints = 0;
  for (const auto &kf : keyFrames)
    for (const auto &e : kf->frames)
      curOptPoints += e.optimizedPoints.size();
  return curOptPoints;
}

bool DsoSystem::doNeedKf(PreKeyFrame *lastFrame) {
  int curTracked = baseFrame().trackedFrames.size();
  return curTracked > 0 &&
         curTracked % settings.keyFrameDist() == settings.keyFrameDist() - 1;
}

void DsoSystem::addFrameTrackerObserver(FrameTrackerObserver *observer) {
  observers.frameTracker.push_back(observer);
}

void DsoSystem::marginalizeFrames() {
  if (settings.disableMarginalization ||
      keyFrames.size() <= settings.maxKeyFrames())
    return;

  int count = static_cast<int>(keyFrames.size()) - settings.maxKeyFrames();
  std::vector<const KeyFrame *> marginalized(count);
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
    obs->keyFramesMarginalized(marginalized.data(), count);

  keyFrames.erase(keyFrames.begin(), keyFrames.begin() + count);

  LOG(INFO) << "marginalized " << count << " frame(s)";
}

void DsoSystem::activateOptimizedDist() {
  std::vector<const KeyFrame *> kfPtrs = getKeyFrames();
  int numAvail = 0;
  for (auto kf : kfPtrs)
    for (const auto &f : kf->frames)
      for (const auto &p : f.immaturePoints)
        if (p.isReady())
          numAvail++;
  VLOG(1) << "num avail = " << numAvail;

  Reprojector<OptimizedPoint> optimizedReprojector(kfPtrs.data(), kfPtrs.size(),
                                                   baseFrame().thisToWorld(),
                                                   settings.depth, PH);
  DepthedPoints optimizedReproj = optimizedReprojector.reprojectDepthed();

  DistanceMap distMap(cam, optimizedReproj.points.data(), settings.distanceMap);

  Reprojector<ImmaturePoint> immatureReprojector(kfPtrs.data(), kfPtrs.size(),
                                                 baseFrame().thisToWorld(),
                                                 settings.depth, PH);
  StdVector<Reprojection> immatureReprojections =
      immatureReprojector.reproject();

  std::vector<StdVector<Vec2>> readyProjected(cam->bundle.size());
  std::vector<std::vector<ImmaturePoint *>> readyPtrs(cam->bundle.size());
  std::vector<std::vector<int>> readyInds(cam->bundle.size());
  for (int camInd = 0; camInd < readyProjected.size(); ++camInd) {
    readyProjected[camInd].reserve(immatureReprojections.size());
    readyPtrs[camInd].reserve(immatureReprojections.size());
    readyInds[camInd].reserve(immatureReprojections.size());
  }
  for (const auto &reproj : immatureReprojections) {
    ImmaturePoint &p = keyFrames[reproj.hostInd]
                           ->frames[reproj.hostCamInd]
                           .immaturePoints[reproj.pointInd];
    if (p.isReady()) {
      readyProjected[reproj.targetCamInd].push_back(reproj.reprojected);
      readyPtrs[reproj.targetCamInd].push_back(&p);
      readyInds[reproj.targetCamInd].push_back(reproj.pointInd);
    }
  }

  int curOptPoints = totalOptimized();
  int pointsNeeded = settings.maxOptimizedPoints() - curOptPoints;

  for (int camInd = 0; camInd < readyProjected.size(); ++camInd)
    VLOG(1) << "Available reprojected on cam #" << camInd << ": "
            << readyProjected[camInd].size() << std::endl;

  std::vector<std::vector<int>> chosenInds(cam->bundle.size());
  int chosenCount =
      distMap.choose(readyProjected.data(), pointsNeeded, chosenInds.data());

  std::vector<std::pair<ImmaturePoint *, int>> chosenPoints;
  chosenPoints.reserve(chosenCount);
  for (int ci = 0; ci < cam->bundle.size(); ++ci)
    for (int j : chosenInds[ci])
      chosenPoints.push_back({readyPtrs[ci][j], readyInds[ci][j]});

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
            << ", needed = " << pointsNeeded
            << ", duplicated = " << oldSize - chosenPoints.size() << '\n'
            << "New OptimizedPoint-s = " << chosenPoints.size() << std::endl;

  for (auto [p, ind] : chosenPoints) {
    KeyFrameEntry &entry = *p->host;
    entry.optimizedPoints.emplace_back(*p);
    CHECK(&entry.immaturePoints[ind] == p)
        << "New optimized points selection, invalid pointer";
    if (ind + 1 != entry.immaturePoints.size())
      std::swap(entry.immaturePoints[ind], entry.immaturePoints.back());
    entry.immaturePoints.pop_back();
  }
}

void DsoSystem::activateOptimizedRandom() {
  std::vector<std::tuple<KeyFrame *, int, int>> readyInds;
  for (int ki = 0; ki < keyFrames.size(); ++ki) {
    KeyFrame *kf = keyFrames[ki].get();
    for (int ci = 0; ci < cam->bundle.size(); ++ci)
      for (int i = 0; i < kf->frames[ci].immaturePoints.size(); ++i) {
        const ImmaturePoint &ip = kf->frames[ci].immaturePoints[i];
        if (ip.isReady())
          readyInds.push_back({kf, ci, i});
      }
  }

  int curOptPoints = totalOptimized();
  int pointsNeeded = settings.maxOptimizedPoints() - curOptPoints;
  if (pointsNeeded < readyInds.size()) {
    std::mt19937 mt(FLAGS_deterministic ? 42 : std::random_device()());
    std::shuffle(readyInds.begin(), readyInds.end(), mt);
    readyInds.resize(pointsNeeded);
  }
  std::sort(readyInds.begin(), readyInds.end(),
            [](const auto &a, const auto &b) {
              return std::get<2>(a) > std::get<2>(b);
            });

  for (const auto &[kf, ci, i] : readyInds) {
    ImmaturePoint &ip = kf->frames[ci].immaturePoints[i];
    kf->frames[ci].optimizedPoints.emplace_back(ip);
    std::swap(ip, kf->frames[ci].immaturePoints.back());
    kf->frames[ci].immaturePoints.pop_back();
  }
}

void DsoSystem::activateNewOptimizedPoints() {
  if (settings.useRandomOptimizedChoice)
    activateOptimizedRandom();
  else
    activateOptimizedDist();
}

void DsoSystem::traceOn(const PreKeyFrame &frame) {
  CHECK(cam->bundle.size() == 1) << "Multicamera case is NIY";

  int retByStatus[ImmaturePoint::STATUS_COUNT];
  int totalPoints = 0;
  int becameReady = 0;
  int totalReady = 0;
  std::fill(retByStatus, retByStatus + ImmaturePoint::STATUS_COUNT, 0);
  for (const auto &kf : keyFrames)
    // TODO inds in multicamera
    for (ImmaturePoint &ip : kf->frames[0].immaturePoints) {
      totalPoints++;
      bool wasReady = ip.isReady();
      int status = ip.traceOn(frame.frames[0], ImmaturePoint::NO_DEBUG,
                              pointTracerSettings);
      retByStatus[status]++;
      bool isReady = ip.isReady();
      if (isReady) {
        totalReady++;
        if (!wasReady)
          becameReady++;
      }
    }

  LOG(INFO) << "POINT TRACING:";
  LOG(INFO) << "total points: " << totalPoints;
  LOG(INFO) << "became ready: " << becameReady;
  LOG(INFO) << "total ready: " << totalReady;
  LOG(INFO) << "return by status:";
  for (int s = 0; s < ImmaturePoint::STATUS_COUNT; ++s)
    LOG(INFO) << ImmaturePoint::statusName(ImmaturePoint::TracingStatus(s))
              << ": " << retByStatus[s];
}

template <typename PointT>
FrameTracker::DepthedMultiFrame DsoSystem::getBaseForTrack() const {
  std::vector<const KeyFrame *> kfPtrs;
  kfPtrs.reserve(keyFrames.size());
  for (const auto &keyFrame : keyFrames)
    kfPtrs.push_back(keyFrame.get());
  DepthedPoints depthedPoints =
      Reprojector<PointT>(kfPtrs.data(), kfPtrs.size(),
                          FrameToWorldExtractor()(&baseFrame()), settings.depth,
                          settings.residualPattern.height)
          .reprojectDepthed();

  FrameTracker::DepthedMultiFrame baseForTrack;
  baseForTrack.reserve(cam->bundle.size());
  for (int camInd = 0; camInd < cam->bundle.size(); ++camInd)
    baseForTrack.emplace_back(baseFrame().preKeyFrame->image(camInd),
                              settings.pyramid.levelNum(),
                              depthedPoints.points[camInd].data(),
                              depthedPoints.depths[camInd].data(),
                              depthedPoints.weights[camInd].data(),
                              depthedPoints.points[camInd].size());
  return baseForTrack;
}

std::unique_ptr<BundleAdjuster> DsoSystem::createBundleAdjuster() const {
  switch (settings.optimization.optimizationType) {
  case Settings::Optimization::DISABLED:
    return nullptr;
  case Settings::Optimization::CERES:
    return std::unique_ptr<BundleAdjuster>(new BundleAdjusterCeres());
  default:
    return std::unique_ptr<BundleAdjuster>(new BundleAdjusterSelfMade());
  }
}

void DsoSystem::addMultiFrame(const cv::Mat3b frames[],
                              Timestamp timestamps[]) {
  if (!isInitialized) {
    LOG(INFO) << "put into initializer" << std::endl;

    isInitialized = dsoInitializer->addMultiFrame(frames, timestamps);

    if (isInitialized) {
      LOG(INFO) << "initialization successful" << std::endl;
      DsoInitializer::InitializedVector init = dsoInitializer->initialize();

      for (int i = 0; i < init.size(); ++i) {
        const InitializedFrame &f = init[i];
        keyFrames.emplace_back(
            new KeyFrame(f, cam, preprocessor, i, pixelSelector.data(),
                         settings.keyFrame, settings.pyramid));
        allFrames.push_back(keyFrames.back().get());
      }

      std::vector<const KeyFrame *> initializedKFs(settings.maxKeyFrames());
      for (int i = 0; i < keyFrames.size(); ++i)
        initializedKFs[i] = keyFrames[i].get();

      for (DsoObserver *obs : observers.dso)
        obs->initialized(initializedKFs.data(), keyFrames.size());

      // BundleAdjusterCeres bundleAdjuster(cam, settings.bundleAdjuster,
      // settings.residualPattern, settings.residualWeighting,
      // settings.intensity, settings.affineLight, settings.threading,
      // settings.depth); for (auto &p : keyFrames)
      // bundleAdjuster.addKeyFrame(&p.second);
      // bundleAdjuster.adjust(settingMaxFirstBAIterations);

      int curPoints = 0;
      for (const auto &kf : keyFrames)
        for (const auto &e : kf->frames)
          curPoints += e.immaturePoints.size();

      std::vector<const KeyFrame *> kfPtrs = getKeyFrames();
      Reprojector<ImmaturePoint> reprojector(kfPtrs.data(), kfPtrs.size(),
                                             baseFrame().thisToWorld(),
                                             settings.depth, PH);
      DepthedPoints reprojected = reprojector.reprojectDepthed();

      FrameTracker::DepthedMultiFrame baseForTrack;
      baseForTrack.reserve(cam->bundle.size());
      for (int i = 0; i < cam->bundle.size(); ++i) {
        std::vector<double> weights(reprojected.points[i].size(), 1.0);
        baseForTrack.emplace_back(
            baseFrame().preKeyFrame->image(i), settings.pyramid.levelNum(),
            reprojected.points[i].data(), reprojected.depths[i].data(),
            weights.data(), reprojected.points[i].size());
      }

      frameTracker = std::unique_ptr<FrameTracker>(new FrameTracker(
          camPyr.data(), baseForTrack, baseFrame(), observers.frameTracker,
          settings.getFrameTrackerSettings()));

      for (DsoObserver *obs : observers.dso)
        obs->newBaseFrame(baseFrame());
    }

    return;
  }

  int globalFrameNum = allFrames.size();

  LOG(INFO) << "add frame #" << globalFrameNum << ", ts[0] = " << timestamps[0]
            << std::endl;

  std::unique_ptr<PreKeyFrame> preKeyFrame(
      new PreKeyFrame(&baseFrame(), cam, preprocessor, frames, globalFrameNum,
                      timestamps, settings.pyramid));

  allFrames.emplace_back(preKeyFrame.get());

  TrackingResult predicted = trackingPredictor->predictAt(
      timestamp(globalFrameNum), baseFrame().preKeyFrame->globalFrameNum);

  preKeyFrame->baseToThisPredicted = predicted.baseToTracked;

  for (DsoObserver *obs : observers.dso)
    obs->newFrame(*preKeyFrame);

  TrackingResult tracked = frameTracker->trackFrame(*preKeyFrame, predicted);

  preKeyFrame->setTracked(tracked);

  SE3 diffPos = predicted.baseToTracked * tracked.baseToTracked.inverse();
  LOG(INFO) << "\ndiff to predicted:\n"
            << "rel trans = "
            << diffPos.translation().norm() /
                   predicted.baseToTracked.translation().norm()
            << "\nrot (deg) = " << diffPos.so3().log().norm() * 180. / M_PI;

  traceOn(*preKeyFrame);

  // for (DsoObserver *obs : observers.dso)
  // obs->pointsTraced ... ;

  bool needNewKf =
      settings.continueChoosingKeyFrames && doNeedKf(preKeyFrame.get());

  if (!needNewKf) {
    baseFrame().trackedFrames.push_back(std::move(preKeyFrame));
  } else {
    SE3 preToWorld = FrameToWorldExtractor()(preKeyFrame.get());

    keyFrames.push_back(std::unique_ptr<KeyFrame>(new KeyFrame(
        std::move(preKeyFrame), pixelSelector.data(), settings.keyFrame)));
    allFrames.back() = keyFrames.back().get();

    SE3 keyToWorld = FrameToWorldExtractor()(keyFrames.back().get());

    SE3 diff = keyToWorld * preToWorld.inverse();

    VLOG(1) << "PKF/KF rel trans diff = "
            << diff.translation().norm() / keyToWorld.translation().norm();
    VLOG(1) << "PKF/KF rot diff = " << diff.log().norm();

    marginalizeFrames();
    activateNewOptimizedPoints();

    for (DsoObserver *obs : observers.dso)
      obs->newBaseFrame(baseFrame());

    std::unique_ptr<BundleAdjuster> bundleAdjuster = createBundleAdjuster();
    if (bundleAdjuster) {
      std::vector<KeyFrame *> kfPtrs(keyFrames.size());
      for (int i = 0; i < keyFrames.size(); ++i)
        kfPtrs[i] = keyFrames[i].get();
      bundleAdjuster->adjust(kfPtrs.data(), kfPtrs.size(),
                             settings.getBundleAdjusterSettings());
    }

    std::vector<const KeyFrame *> kfPtrs = getKeyFrames();
    Reprojector<OptimizedPoint> reprojector(kfPtrs.data(), kfPtrs.size(),
                                            baseFrame().thisToWorld(),
                                            settings.depth, PH);
    DepthedPoints reprojected = reprojector.reprojectDepthed();

    FrameTracker::DepthedMultiFrame baseForTrack;
    baseForTrack.reserve(cam->bundle.size());
    for (int i = 0; i < cam->bundle.size(); ++i) {
      baseForTrack.emplace_back(
          baseFrame().preKeyFrame->image(i), settings.pyramid.levelNum(),
          reprojected.points[i].data(), reprojected.depths[i].data(),
          reprojected.weights[i].data(), reprojected.points[i].size());
    }

    frameTracker = std::unique_ptr<FrameTracker>(new FrameTracker(
        camPyr.data(), baseForTrack, baseFrame(), observers.frameTracker,
        settings.getFrameTrackerSettings()));
  }
}

void DsoSystem::saveSnapshot(const fs::path &snapshotDir) const {
  SnapshotSaver snapshotSaver(snapshotDir,
                              settings.residualPattern.pattern().size());
  std::vector<const KeyFrame *> savedKeyFrames;
  savedKeyFrames.reserve(savedKeyFrames.size());
  for (const auto &keyFrame : keyFrames)
    savedKeyFrames.push_back(keyFrame.get());
  snapshotSaver.save(savedKeyFrames.data(), savedKeyFrames.size());
}

} // namespace mdso
