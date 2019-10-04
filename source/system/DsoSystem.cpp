#include "system/DsoSystem.h"
#include "output/DsoObserver.h"
#include "output/FrameTrackerObserver.h"
#include "system/AffineLightTransform.h"
#include "system/DelaunayDsoInitializer.h"
#include "system/StereoMatcher.h"
#include "system/TrackingPredictorRot.h"
#include "system/TrackingPredictorScrew.h"
#include "util/defs.h"
#include "util/flags.h"
#include "util/geometry.h"
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
                                  new TrackingPredictorRot(this))) {
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

DsoSystem::~DsoSystem() {
  std::vector<const KeyFrame *> lastKeyFrames(keyFrames.size());
  for (int i = 0; i < keyFrames.size(); ++i)
    lastKeyFrames[i] = keyFrames[i].get();
  for (DsoObserver *obs : observers.dso)
    obs->destructed(lastKeyFrames.data(), keyFrames.size());
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
void DsoSystem::projectOntoFrame(int globalFrameNum, Vec2 *points[],
                                 const std::optional<PointT ***> &refs,
                                 const std::optional<int **> &pointIndices,
                                 const std::optional<double **> &depths,
                                 int sizes[]) {
  CHECK(globalFrameNum >= 0 && globalFrameNum < allFrames.size());

  SE3 worldToFrame = bodyToWorld(globalFrameNum).inverse();

  for (int i = 0; i < cam->bundle.size(); ++i)
    sizes[i] = 0;

  for (const auto &kf : keyFrames) {
    SE3 refToFrameBody = worldToFrame * kf->thisToWorld;
    for (int ri = 0; ri < cam->bundle.size(); ++ri) {
      const CameraBundle::CameraEntry &refCam = cam->bundle[ri];
      SE3 temp = refToFrameBody * refCam.thisToBody;
      for (int bi = 0; bi < cam->bundle.size(); ++bi) {
        CameraBundle::CameraEntry &baseCam = cam->bundle[bi];
        SE3 refToFrame = baseCam.bodyToThis * temp;
        auto &pointVec = getPoints<PointT>(kf->frames[ri]);
        for (int pi = 0; pi < pointVec.size(); ++pi) {
          PointT &pnt = pointVec[pi];
          if (pnt.state != PointT::ACTIVE)
            continue;

          Vec3 baseDir = refToFrame * (getDepth(pnt) * pnt.dir);
          Vec3 baseDirNormalized = baseDir.normalized();
          if (baseDirNormalized[2] < baseCam.cam.getMinZ())
            continue;
          Vec2 proj = baseCam.cam.map(baseDir);
          if (baseCam.cam.isOnImage(proj, PH)) {
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

int DsoSystem::trajectorySize() const {
  if (allFrames.size() > 0 &&
      std::holds_alternative<PreKeyFrame *>(allFrames.back())) {
    PreKeyFrame *last = std::get<PreKeyFrame *>(allFrames.back());
    return last->wasTracked() ? allFrames.size() : int(allFrames.size()) - 1;
  }
  return allFrames.size() - 1;
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
  SE3 operator()(MarginalizedKeyFrame *f) { return f->thisToWorld; }
  SE3 operator()(MarginalizedPreKeyFrame *f) {
    return f->baseFrame->thisToWorld * f->baseToThis.inverse();
  }
  SE3 operator()(KeyFrame *f) { return f->thisToWorld; }
  SE3 operator()(PreKeyFrame *f) {
    return f->baseFrame->thisToWorld * f->baseToThis().inverse();
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

int DsoSystem::totalOptimized() const {
  int curOptPoints = 0;
  for (const auto &kf : keyFrames)
    for (const auto &e : kf->frames)
      curOptPoints += e.optimizedPoints.size();
  return curOptPoints;
}

bool DsoSystem::doNeedKf(PreKeyFrame *lastFrame) {
  return allFrames.size() % settings.keyFrameDist() ==
         (settings.keyFrameDist() - 1);
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
  std::vector<StdVector<Vec2>> optPoints(cam->bundle.size());
  std::vector<Vec2 *> optPointsPtrs(cam->bundle.size());
  std::vector<int> optSizes(cam->bundle.size(), 0);
  for (int i = 0; i < cam->bundle.size(); ++i) {
    optPoints[i].resize(settings.maxOptimizedPoints());
    optPointsPtrs[i] = optPoints[i].data();
  }

  projectOntoFrame<OptimizedPoint>(baseFrame().preKeyFrame->globalFrameNum,
                                   optPointsPtrs.data(), std::nullopt,
                                   std::nullopt, std::nullopt, optSizes.data());
  for (int i = 0; i < cam->bundle.size(); ++i)
    optPoints[i].resize(optSizes[i]);

  DistanceMap distMap(cam, optPoints.data(), settings.distanceMap);

  std::vector<StdVector<Vec2>> projImm(cam->bundle.size());
  std::vector<std::vector<ImmaturePoint *>> refsImm(cam->bundle.size());
  std::vector<std::vector<int>> pointIndicesImm(cam->bundle.size());
  std::vector<Vec2 *> projImmPtrs(cam->bundle.size());
  std::vector<ImmaturePoint **> refsImmPtrs(cam->bundle.size());
  std::vector<int *> pointIndicesImmPtrs(cam->bundle.size());
  std::vector<int> immSizes(cam->bundle.size(), 0);
  int toResize =
      settings.maxKeyFrames() * settings.keyFrame.immaturePointsNum();
  for (int i = 0; i < cam->bundle.size(); ++i) {
    projImm[i].resize(toResize);
    refsImm[i].resize(toResize);
    pointIndicesImm[i].resize(toResize);
    projImmPtrs[i] = projImm[i].data();
    refsImmPtrs[i] = refsImm[i].data();
    pointIndicesImmPtrs[i] = pointIndicesImm[i].data();
  }

  projectOntoFrame<ImmaturePoint>(
      baseFrame().preKeyFrame->globalFrameNum, projImmPtrs.data(),
      std::make_optional(refsImmPtrs.data()),
      std::make_optional(pointIndicesImmPtrs.data()), std::nullopt,
      immSizes.data());
  std::vector<std::vector<int>> chosenInds(cam->bundle.size());
  for (int i = 0; i < cam->bundle.size(); ++i) {
    int areReady = 0;
    for (int j = 0; j < immSizes[i]; ++j)
      if (refsImm[i][j]->isReady()) {
        projImm[i][areReady] = projImm[i][j];
        refsImm[i][areReady] = refsImm[i][j];
        pointIndicesImm[i][areReady] = pointIndicesImm[i][j];
        areReady++;
      }

    projImm[i].resize(areReady);
    refsImm[i].resize(areReady);
    pointIndicesImm[i].resize(areReady);
    immSizes[i] = areReady;
    chosenInds[i].reserve(areReady);
  }

  int curOptPoints = totalOptimized();
  int pointsNeeded = settings.maxOptimizedPoints() - curOptPoints;

  int chosenCount =
      distMap.choose(projImm.data(), pointsNeeded, chosenInds.data());

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
  if (FLAGS_use_random_optimized_choice)
    activateOptimizedRandom();
  else
    activateOptimizedDist();
}

void DsoSystem::traceOn(const PreKeyFrame &frame) {
  CHECK(cam->bundle.size() == 1) << "Multicamera case is NIY";

  int retByStatus[ImmaturePoint::STATUS_COUNT];
  int totalPoints = 0;
  int becameReady = 0;
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
      if (isReady && !wasReady)
        becameReady++;
    }

  LOG(INFO) << "POINT TRACING:";
  LOG(INFO) << "total points: " << totalPoints;
  LOG(INFO) << "became ready: " << becameReady;
  LOG(INFO) << "return by status:";
  for (int s = 0; s < ImmaturePoint::STATUS_COUNT; ++s)
    LOG(INFO) << ImmaturePoint::statusName(ImmaturePoint::TracingStatus(s))
              << ": " << retByStatus[s];
}

void DsoSystem::addMultiFrame(const cv::Mat frames[], Timestamp timestamps[]) {
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
      std::vector<StdVector<Vec2>> points(cam->bundle.size());
      std::vector<std::vector<double>> depths(cam->bundle.size());
      std::vector<int> sizes(cam->bundle.size());
      std::vector<Vec2 *> pointPtrs(cam->bundle.size());
      std::vector<double *> depthPtrs(cam->bundle.size());
      for (int i = 0; i < cam->bundle.size(); ++i) {
        points[i].resize(curPoints);
        depths[i].resize(curPoints);
        pointPtrs[i] = points[i].data();
        depthPtrs[i] = depths[i].data();
      }

      projectOntoFrame<ImmaturePoint>(
          baseFrame().preKeyFrame->globalFrameNum, pointPtrs.data(),
          std::nullopt, std::nullopt, std::make_optional(depthPtrs.data()),
          sizes.data());
      FrameTracker::DepthedMultiFrame baseForTrack;
      baseForTrack.reserve(cam->bundle.size());
      for (int i = 0; i < cam->bundle.size(); ++i) {
        points[i].resize(sizes[i]);
        depths[i].resize(sizes[i]);
        std::vector<double> weights(points[i].size(), 1);
        baseForTrack.emplace_back(baseFrame().preKeyFrame->image(i),
                                  settings.pyramid.levelNum(), points[i].data(),
                                  depths[i].data(), weights.data(),
                                  points[i].size());
      }

      VLOG(1) << "baseForTrack size = " << baseForTrack.size()
              << " imgages = " << baseForTrack[0].images.size()
              << "Depths = " << baseForTrack[0].depths.size()
              << "depths[0].size = " << baseForTrack[0].depths[0].size();

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

  allFrames.push_back(preKeyFrame.get());

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

    if (settings.bundleAdjuster.runBA) {
      std::vector<KeyFrame *> kfPtrs(keyFrames.size());
      for (int i = 0; i < keyFrames.size(); ++i)
        kfPtrs[i] = keyFrames[i].get();
      BundleAdjuster bundleAdjuster(cam, kfPtrs.data(), keyFrames.size(),
                                    settings.getBundleAdjusterSettings());
      bundleAdjuster.adjust(settings.bundleAdjuster.maxIterations);
    }

    std::vector<StdVector<Vec2>> points(cam->bundle.size());
    std::vector<std::vector<double>> depths(cam->bundle.size());
    std::vector<std::vector<OptimizedPoint *>> refs(cam->bundle.size());
    std::vector<Vec2 *> pointsPtrs(cam->bundle.size());
    std::vector<double *> depthsPtrs(cam->bundle.size());
    std::vector<OptimizedPoint **> refsPtrs(cam->bundle.size());
    std::vector<int> sizes(cam->bundle.size());
    for (int i = 0; i < cam->bundle.size(); ++i) {
      points[i].resize(settings.maxOptimizedPoints());
      refs[i].resize(settings.maxOptimizedPoints());
      depths[i].resize(settings.maxOptimizedPoints());
      pointsPtrs[i] = points[i].data();
      refsPtrs[i] = refs[i].data();
      depthsPtrs[i] = depths[i].data();
    }

    projectOntoFrame<OptimizedPoint>(
        baseFrame().preKeyFrame->globalFrameNum, pointsPtrs.data(),
        std::make_optional(refsPtrs.data()), std::nullopt,
        std::make_optional(depthsPtrs.data()), sizes.data());

    FrameTracker::DepthedMultiFrame baseForTrack;
    baseForTrack.reserve(cam->bundle.size());
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

    frameTracker = std::unique_ptr<FrameTracker>(new FrameTracker(
        camPyr.data(), baseForTrack, baseFrame(), observers.frameTracker,
        settings.getFrameTrackerSettings()));
  }
}

} // namespace mdso
