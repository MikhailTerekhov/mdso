#include "system/DsoSystem.h"
#include "system/AffineLightTransform.h"
#include "system/DelaunayDsoInitializer.h"
#include "system/StereoMatcher.h"
#include "util/defs.h"
#include "util/geometry.h"
#include "util/settings.h"
#include <glog/logging.h>

namespace fishdso {

DsoSystem::DsoSystem(CameraModel *cam)
    : lastInitialized(nullptr), scaleGTToOur(1.0), cam(cam),
      camPyr(cam->camPyr()), pixelSelector(),
      dsoInitializer(std::unique_ptr<DsoInitializer>(new DelaunayDsoInitializer(
          this, cam, &pixelSelector, DelaunayDsoInitializer::SPARSE_DEPTHS))),
      isInitialized(false), lastTrackRmse(INF), firstFrameNum(-1) {
  LOG(INFO) << "create DsoSystem" << std::endl;

  if (FLAGS_write_files)
    cloudHolder.emplace(FLAGS_output_directory + "/points.ply");
}

DsoSystem::~DsoSystem() {}

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

EIGEN_STRONG_INLINE SE3 predictInternal(int prevFramesSkipped,
                                        const SE3 &worldToBaseKf,
                                        const SE3 &worldToLbo,
                                        const SE3 &worldToLast) {
  return FLAGS_predict_using_screw
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

void DsoSystem::addGroundTruthPose(int ind, const SE3 &worldToThat) {
  SE3 newPose = worldToThat;
  newPose.translation() *= scaleGTToOur;
  worldToFrameGT[ind] = newPose * gtToOur;
}

void DsoSystem::fillRemainedHistory() {
  if (FLAGS_write_files)
    saveOldKfs(keyFrames.size());
}

void DsoSystem::alignGTPoses() {
  SE3 worldToFirst = keyFrames.begin()->second.preKeyFrame->worldToThis;
  SE3 worldToLast = keyFrames.rbegin()->second.preKeyFrame->worldToThis;
  SE3 fToL = worldToLast * worldToFirst.inverse();
  int firstNum = keyFrames.begin()->second.preKeyFrame->globalFrameNum;
  int lastNum = keyFrames.rbegin()->second.preKeyFrame->globalFrameNum;
  auto itF = worldToFrameGT.find(firstNum);
  auto itL = worldToFrameGT.find(lastNum);
  if (itF == worldToFrameGT.end() || itL == worldToFrameGT.end()) {
    LOG(WARNING) << "Could not align GT poses as some frame poses miss (#"
                 << firstNum << " or #" << lastNum << ")" << std::endl;
    return;
  }

  SE3 worldToFirstGT = itF->second;
  SE3 worldToLastGT = itL->second;
  SE3 fToLGT = worldToLastGT * worldToFirstGT.inverse();
  if (fToLGT.translation().norm() < 1e-3 || fToL.translation().norm() < 1e-3) {
    LOG(WARNING) << "Could not align GT poses as motion is too static"
                 << std::endl;
    return;
  }

  scaleGTToOur = fToL.translation().norm() / fToLGT.translation().norm();
  SE3 worldToFirstGTScaled = worldToFirstGT;
  worldToFirstGTScaled.translation() *= scaleGTToOur;
  gtToOur = worldToFirstGTScaled.inverse() * worldToFirst;

  for (auto &pose : worldToFrameGT) {
    pose.second.translation() *= scaleGTToOur;
    pose.second = pose.second * gtToOur;
  }
}

bool DsoSystem::doNeedKf(PreKeyFrame *lastFrame) {
  int shift = lastFrame->globalFrameNum - firstFrameNum;
  return shift > 0 && shift % 10 == 0;
}

void DsoSystem::saveOldKfs(int count) {
  CHECK(cloudHolder);

  std::ofstream posesOfs(FLAGS_output_directory + "/tracked_pos.txt",
                         std::ios_base::app);
  std::ofstream posesGtOfs(FLAGS_output_directory + "/ground_truth_pos.txt",
                           std::ios_base::app);

  auto it = keyFrames.begin();
  int firstFrameNum = it->first;
  for (int i = 0; i < count; ++i) {
    const KeyFrame &kf = it->second;
    it++;

    std::vector<Vec3> points;
    std::vector<cv::Vec3b> colors;

    for (const auto &op : kf.optimizedPoints) {
      points.push_back(kf.preKeyFrame->worldToThis.inverse() *
                       (op->depth() * cam->unmap(op->p).normalized()));
      colors.push_back(
          kf.preKeyFrame->frameColored.at<cv::Vec3b>(toCvPoint(op->p)));
    }
    for (const auto &ip : kf.immaturePoints) {
      if (ip->numTraced > 0) {
        points.push_back(kf.preKeyFrame->worldToThis.inverse() *
                         (ip->depth * cam->unmap(ip->p).normalized()));
        colors.push_back(
            kf.preKeyFrame->frameColored.at<cv::Vec3b>(toCvPoint(ip->p)));
      }
    }

    int kfnum = kf.preKeyFrame->globalFrameNum;
    std::ofstream kfOut(FLAGS_output_directory + "/kf" + std::to_string(kfnum) +
                        ".ply");
    printInPly(kfOut, points, colors);
    kfOut.close();
    cloudHolder.value().putPoints(points, colors);
  }

  auto poseItEnd = worldToFrame.end();
  auto poseGtItEnd = worldToFrameGT.end();
  if (it != keyFrames.end()) {
    poseItEnd = worldToFrame.lower_bound(it->first);
    poseGtItEnd = worldToFrameGT.lower_bound(it->first);
  }

  for (auto poseIt = worldToFrame.find(firstFrameNum); poseIt != poseItEnd;
       ++poseIt) {
    posesOfs << poseIt->first << ' ';
    putMotion(posesOfs, poseIt->second);
    posesOfs << std::endl;
  }

  for (auto poseGtIt = worldToFrameGT.find(firstFrameNum);
       poseGtIt != poseGtItEnd; ++poseGtIt) {
    posesGtOfs << poseGtIt->first << ' ';
    putMotion(posesGtOfs, poseGtIt->second);
    posesGtOfs << std::endl;
  }

  cloudHolder.value().updatePointCount();

  posesOfs.close();
  posesGtOfs.close();
}

void DsoSystem::marginalizeFrames() {
  if (keyFrames.size() > settingMaxKeyFrames) {
    int count = static_cast<int>(keyFrames.size()) - settingMaxKeyFrames;
    if (FLAGS_write_files)
      saveOldKfs(count);
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
  int pointsNeeded = settingMaxOptimizedPoints - curOptPoints;
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
  return frameTracker->lastRmse > lastTrackRmse * FLAGS_track_fail_factor;
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

      if (FLAGS_perform_tracking_check_GT)
        alignGTPoses();

      // BundleAdjuster bundleAdjuster(cam);
      // for (auto &p : keyFrames)
      // bundleAdjuster.addKeyFrame(&p.second);

      // if (FLAGS_write_files) {
      // std::ofstream ofsBeforeAdjust(FLAGS_output_directory +
      // "/before_adjust.ply");
      // printLastKfInPly(ofsBeforeAdjust);
      // }

      // bundleAdjuster.adjust(settingMaxFirstBAIterations);

      if (FLAGS_switch_first_motion_to_GT || FLAGS_gt_poses) {
        SE3 worldToSecondKfGT =
            worldToFrameGT[keyFrames.rbegin()
                               ->second.preKeyFrame->globalFrameNum];
        keyFrames.rbegin()->second.preKeyFrame->worldToThis = worldToSecondKfGT;
        worldToFrame[keyFrames.rbegin()->second.preKeyFrame->globalFrameNum] =
            worldToSecondKfGT;
      }

      if (FLAGS_write_files) {
        std::ofstream ofsAfterAdjust(FLAGS_output_directory + "/init.ply");
        printLastKfInPly(ofsAfterAdjust);
      }

      StdVector<Vec2> points;
      std::vector<double> depths;
      projectOntoBaseKf<ImmaturePoint>(&points, &depths, nullptr, nullptr);

      std::vector<double> weights(points.size(), 1.0);

      std::unique_ptr<DepthedImagePyramid> initialTrack(new DepthedImagePyramid(
          baseKeyFrame().preKeyFrame->frame(), points, depths, weights));

      frameTracker = std::unique_ptr<FrameTracker>(
          new FrameTracker(camPyr, std::move(initialTrack)));

      lastInitialized = &keyFrames.rbegin()->second;

      firstFrameNum = lastInitialized->preKeyFrame->globalFrameNum;

      return lastInitialized->preKeyFrame;
    }

    return nullptr;
  }

  std::shared_ptr<PreKeyFrame> preKeyFrame(
      new PreKeyFrame(cam, frame, globalFrameNum));

  SE3 baseKfToCur;
  AffineLightTransform<double> lightBaseKfToCur;

  SE3 purePredicted = purePredictBaseKfToCur();
  SE3 predicted = predictBaseKfToCur();

  LOG(INFO) << "start tracking this frame" << std::endl;

  std::tie(baseKfToCur, lightBaseKfToCur) = frameTracker->trackFrame(
      ImagePyramid(preKeyFrame->frame()), predicted, lightKfToLast);

  LOG(INFO) << "tracking ended" << std::endl;

  PreKeyFrame *baseKf = baseKeyFrame().preKeyFrame.get();
  preKeyFrame->lightWorldToThis = lightBaseKfToCur * baseKf->lightWorldToThis;

  LOG(INFO) << "aff light: (fnum=" << preKeyFrame->globalFrameNum << ")\n"
            << preKeyFrame->lightWorldToThis << std::endl;

  worldToFrame[globalFrameNum] = baseKfToCur * baseKf->worldToThis;
  worldToFramePredict[globalFrameNum] = purePredicted * baseKf->worldToThis;
  preKeyFrame->worldToThis = worldToFrame[globalFrameNum];

  if (FLAGS_gt_poses) {
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

  if (FLAGS_perform_tracking_check_GT) {
    LOG(INFO) << "perform comparison to ground truth" << std::endl;
    checkLastTrackedGT(preKeyFrame.get());
  } else if (FLAGS_perform_tracking_check_stereo) {
    LOG(INFO) << "perform stereo check" << std::endl;
    checkLastTrackedStereo(preKeyFrame.get());
  }

  int totalActive = 0, totalGood = 0;
  constexpr int maxTraced = 5;
  std::array<int, maxTraced> numTraced;
  numTraced.fill(0);
  std::array<int, settingPyrLevels> numOnLevel;
  numOnLevel.fill(0);
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
  for (int i = 0; i < settingPyrLevels; ++i)
    std::cout << numOnLevel[i] << ' ';
  std::cout << std::endl;

  if (FLAGS_write_files || FLAGS_show_debug_image) {
    cv::Mat3b debugImg = drawDebugImage(preKeyFrame);
    if (FLAGS_write_files) {
      std::string filename =
          "/frame" + std::to_string(preKeyFrame->globalFrameNum) + ".jpg";
      cv::imwrite(FLAGS_debug_img_dir + filename, debugImg);
    }
    if (FLAGS_show_debug_image) {
      cv::imshow("debug", debugImg);
      cv::waitKey(1);
    }
  }
  if (FLAGS_write_files || FLAGS_show_track_res) {
    cv::Mat3b trackLevels =
        drawLeveled(frameTracker->residualsImg, settingPyrLevels,
                    cam->getWidth(), cam->getHeight());
    if (FLAGS_write_files) {
      std::string filename =
          "/frame" + std::to_string(preKeyFrame->globalFrameNum) + ".jpg";
      cv::imwrite(FLAGS_track_img_dir + filename, trackLevels);
    }
    if (FLAGS_show_track_res) {
      cv::imshow("tracking residuals", trackLevels);
      cv::waitKey(1);
    }
  }

  if (FLAGS_continue_choosing_keyframes && doNeedKf(preKeyFrame.get())) {
    int kfNum = preKeyFrame->globalFrameNum;
    keyFrames.insert(
        std::pair<int, KeyFrame>(kfNum, KeyFrame(preKeyFrame, pixelSelector)));

    marginalizeFrames();
    activateNewOptimizedPoints();

    if (FLAGS_run_ba) {
      BundleAdjuster bundleAdjuster(cam);
      for (auto &kfp : keyFrames)
        bundleAdjuster.addKeyFrame(&kfp.second);
      bundleAdjuster.adjust(settingMaxBAIterations);
    }

    StdVector<Vec2> points;
    std::vector<double> depths;
    std::vector<OptimizedPoint *> refs;
    projectOntoBaseKf<OptimizedPoint>(&points, &depths, &refs, nullptr);
    std::vector<double> weights(points.size());
    for (int i = 0; i < points.size(); ++i)
      weights[i] = 1.0 / refs[i]->stddev;
    std::unique_ptr<DepthedImagePyramid> baseForTrack(new DepthedImagePyramid(
        baseKeyFrame().preKeyFrame->frame(), points, depths, weights));

    for (int i = 0; i < points.size(); ++i) {
      for (int pl = 0; pl < settingPyrLevels; ++pl) {
        cv::Point p = toCvPoint(points[i]);
        if (baseForTrack->depths[pl](p / (1 << pl)) <= 0) {
          std::cout << "pl=" << pl << " p=" << p << " psh=" << p / (1 << pl)
                    << " i=" << i << " d=" << depths[i] << " w=" << weights[i]
                    << " porig=" << points[i] << std::endl;
        }
      }
    }

    if (FLAGS_show_track_base) {
      cv::Mat3b trackImg = baseForTrack->draw();
      cv::imshow("tracking base", trackImg);
      int fnum = baseKeyFrame().preKeyFrame->globalFrameNum;
      if (FLAGS_write_files)
        cv::imwrite(FLAGS_output_directory + "/frame" + std::to_string(fnum) +
                        ".jpg",
                    trackImg);
    }

    frameTracker = std::unique_ptr<FrameTracker>(
        new FrameTracker(camPyr, std::move(baseForTrack)));
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

void DsoSystem::printLastKfInPly(std::ostream &out) {
  StdVector<std::pair<Vec2, double>> points;
  points.reserve(lastKeyFrame().optimizedPoints.size() +
                 lastKeyFrame().immaturePoints.size());
  for (const auto &op : lastKeyFrame().optimizedPoints) {
    double d = op->depth();
    if (!std::isfinite(d) || d > 1e3)
      continue;

    points.push_back({op->p, d});
  }

  for (const auto &ip : lastKeyFrame().immaturePoints) {
    double d = ip->depth;
    if (!std::isfinite(d) || d > 1e3)
      continue;

    points.push_back({ip->p, d});
  }

  out.precision(15);
  out << R"__(ply
format ascii 1.0
element vertex )__"
      << points.size() << R"__(
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
)__";

  for (const auto &p : points) {
    Vec3 pos = cam->unmap(p.first).normalized() * p.second;
    out << pos[0] << ' ' << pos[1] << ' ' << pos[2] << ' ';
    cv::Vec3b color = lastKeyFrame().preKeyFrame->frameColored.at<cv::Vec3b>(
        toCvPoint(p.first));
    out << int(color[2]) << ' ' << int(color[1]) << ' ' << int(color[0])
        << std::endl;
  }
}

void DsoSystem::printMotionInfo(std::ostream &out,
                                const StdMap<int, SE3> &motions) {
  out.precision(15);
  for (auto p : motions) {
    out << p.first << ' ';
    putMotion(out, p.second);
    out << std::endl;
  }
}

void DsoSystem::printTrackingInfo(std::ostream &out) {
  printMotionInfo(out, worldToFrame);
}

void DsoSystem::printPredictionInfo(std::ostream &out) {
  printMotionInfo(out, worldToFramePredict);
}

void DsoSystem::printMatcherInfo(std::ostream &out) {
  printMotionInfo(out, worldToFrameMatched);
}

void DsoSystem::printGroundTruthInfo(std::ostream &out) {
  printMotionInfo(out, worldToFrameGT);
}

cv::Mat3b
DsoSystem::drawDebugImage(const std::shared_ptr<PreKeyFrame> &lastFrame) {
  int w = cam->getWidth(), h = cam->getHeight();
  int s = FLAGS_rel_point_size * (w + h) / 2;

  cv::Mat3b base = cvtBgrToGray3(baseKeyFrame().preKeyFrame->frameColored);
  StdVector<Vec2> immPt;
  std::vector<double> immD;
  std::vector<ImmaturePoint *> immRef;
  projectOntoBaseKf<ImmaturePoint>(&immPt, &immD, &immRef, nullptr);
  StdVector<Vec2> optPt;
  std::vector<double> optD;
  std::vector<OptimizedPoint *> optRef;
  projectOntoBaseKf<OptimizedPoint>(&optPt, &optD, &optRef, nullptr);

  cv::Mat3b depths = base.clone();
  for (int i = 0; i < immPt.size(); ++i)
    if (immRef[i]->numTraced > 0)
      putSquare(depths, toCvPoint(immPt[i]), s,
                depthCol(immD[i], minDepthCol, maxDepthCol), cv::FILLED);
  for (int i = 0; i < optPt.size(); ++i)
    putSquare(depths, toCvPoint(optPt[i]), s,
              depthCol(optD[i], minDepthCol, maxDepthCol), cv::FILLED);

  cv::Mat3b usefulImg = base.clone();
  SE3 baseToLast = lastFrame->worldToThis *
                   baseKeyFrame().preKeyFrame->worldToThis.inverse();
  for (int i = 0; i < optPt.size(); ++i) {
    Vec3 p = optD[i] * cam->unmap(optPt[i]).normalized();
    Vec2 reproj = cam->map(baseToLast * p);
    cv::Scalar col =
        cam->isOnImage(reproj, settingResidualPatternSize) ? CV_GREEN : CV_RED;
    putSquare(usefulImg, toCvPoint(optPt[i]), s, col, cv::FILLED);
  }

  cv::Mat3b stddevs = base.clone();
  for (int i = 0; i < immPt.size(); ++i) {
    double dev = immRef[i]->stddev;
    if (immRef[i]->numTraced > 0)
      putSquare(stddevs, toCvPoint(immPt[i]), s,
                depthCol(dev, MIN_STDDEV, FLAGS_debug_max_stddev), cv::FILLED);
  }
  for (int i = 0; i < optPt.size(); ++i) {
    double dev = optRef[i]->stddev;
    putSquare(stddevs, toCvPoint(optPt[i]), s,
              depthCol(dev, MIN_STDDEV, FLAGS_debug_max_stddev), cv::FILLED);
  }

  cv::Mat3b row1, row2, resultBig, result;
  cv::hconcat(depths, usefulImg, row1);
  cv::hconcat(stddevs, frameTracker->residualsImg[0], row2);
  cv::vconcat(row1, row2, resultBig);
  int newh = double(resultBig.rows) / resultBig.cols * FLAGS_debug_width;
  cv::resize(resultBig, result, cv::Size(FLAGS_debug_width, newh));
  return result;
}

} // namespace fishdso
