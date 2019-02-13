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
    : cam(cam), camPyr(cam->camPyr()), pixelSelector(),
      dsoInitializer(std::unique_ptr<DsoInitializer>(new DelaunayDsoInitializer(
          this, cam, &pixelSelector, DelaunayDsoInitializer::SPARSE_DEPTHS))),
      isInitialized(false), lastInitialized(nullptr),
      adaptiveBlockSize(settingInitialAdaptiveBlockSize), scaleGTToOur(1.0) {
  LOG(INFO) << "create DsoSystem" << std::endl;
}

DsoSystem::~DsoSystem() {
  if (FLAGS_write_files) {
    std::ofstream ofsTracked(FLAGS_output_directory + "/tracked_pos.txt");
    printTrackingInfo(ofsTracked);
    std::ofstream ofsPredicted(FLAGS_output_directory + "/predicted_pos.txt");
    printPredictionInfo(ofsPredicted);

    if (FLAGS_perform_tracking_check_GT) {
      std::ofstream ofsGT(FLAGS_output_directory + "/ground_truth_pos.txt");
      printGroundTruthInfo(ofsGT);
    } else if (FLAGS_perform_tracking_check_stereo) {
      std::ofstream ofsMatched(FLAGS_output_directory + "/matched_pos.txt");
      printMatcherInfo(ofsMatched);
    }
  }
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
  for (const auto &kfp : keyFrames) {
    auto &kf = kfp.second;
    for (const auto &op : kf.optimizedPoints) {
      pointHistory.push_back(kf.preKeyFrame->worldToThis.inverse() *
                             (op->depth() * cam->unmap(op->p)));
      pointHistoryCol.push_back(
          kf.preKeyFrame->frameColored.at<cv::Vec3b>(toCvPoint(op->p)));
    }
    for (const auto &ip : kf.immaturePoints) {
      if (ip->state == ImmaturePoint::ACTIVE &&
          ip->stddev < settingMaxOptimizedStddev * 1.2) {
        pointHistory.push_back(kf.preKeyFrame->worldToThis.inverse() *
                               (ip->depth * cam->unmap(ip->p)));
        pointHistoryCol.push_back(
            kf.preKeyFrame->frameColored.at<cv::Vec3b>(toCvPoint(ip->p)));
      }
    }
  }
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

bool DsoSystem::checkNeedKf(PreKeyFrame *lastFrame) {
  int shift = lastFrame->globalFrameNum - frameHistory[0]->globalFrameNum;
  return shift > 0 && shift % 10 == 0;
}

void DsoSystem::marginalizeFrames() {
  if (keyFrames.size() > settingMaxKeyFrames)
    for (int i = 0;
         i < static_cast<int>(keyFrames.size()) - settingMaxKeyFrames; ++i) {
      auto &kf = keyFrames.begin()->second;
      for (const auto &op : kf.optimizedPoints) {
        pointHistory.push_back(kf.preKeyFrame->worldToThis.inverse() *
                               (op->depth() * cam->unmap(op->p)));
        pointHistoryCol.push_back(
            kf.preKeyFrame->frameColored.at<cv::Vec3b>(toCvPoint(op->p)));
      }
      for (const auto &ip : kf.immaturePoints) {
        if (ip->state == ImmaturePoint::ACTIVE &&
            ip->stddev < settingMaxOptimizedStddev * 1.2) {
          pointHistory.push_back(kf.preKeyFrame->worldToThis.inverse() *
                                 (ip->depth * cam->unmap(ip->p)));
          pointHistoryCol.push_back(
              kf.preKeyFrame->frameColored.at<cv::Vec3b>(toCvPoint(ip->p)));
        }
      }
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

    if (immRefs[oldIt]->state == ImmaturePoint::ACTIVE &&
        immRefs[oldIt]->stddev < settingMaxOptimizedStddev) {
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

DEFINE_bool(gt_poses, false,
            "Fix all poses of frames to GT. Enabled to check if tracing got "
            "problems on its own, or these problems lie in poor tracking.");

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

      if (FLAGS_switch_first_motion_to_GT) {
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

      int pointsTotal = std::accumulate(
          keyFrames.begin(), keyFrames.end(), 0, [](int acc, const auto &kfp) {
            return acc + kfp.second.immaturePoints.size();
          });
      StdVector<Vec2> points;
      std::vector<double> depths;
      projectOntoBaseKf<ImmaturePoint>(&points, &depths, nullptr, nullptr);

      std::vector<double> weights(points.size(), 1.0);

      std::unique_ptr<DepthedImagePyramid> initialTrack(new DepthedImagePyramid(
          baseKeyFrame().preKeyFrame->frame(), points, depths, weights));

      frameTracker = std::unique_ptr<FrameTracker>(
          new FrameTracker(camPyr, std::move(initialTrack)));

      lastInitialized = &keyFrames.rbegin()->second;
      return lastInitialized->preKeyFrame;
    }

    return nullptr;
  }

  std::shared_ptr<PreKeyFrame> preKeyFrame(
      new PreKeyFrame(cam, frame, globalFrameNum));

  frameHistory.push_back(preKeyFrame);

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
  for (const auto &kfp : keyFrames)
    for (auto &ip : kfp.second.immaturePoints) {
      ip->traceOn(*preKeyFrame, ImmaturePoint::NO_DEBUG);
      if (ip->state == ImmaturePoint::ACTIVE)
        totalActive++;
      if (ip->state == ImmaturePoint::ACTIVE &&
          ip->stddev < settingMaxOptimizedStddev) {
        totalGood++;
      }
    }
  std::cout << "active = " << totalActive << " good = " << totalGood
            << std::endl;

  if (FLAGS_continue_choosing_keyframes && checkNeedKf(preKeyFrame.get())) {
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

    if (FLAGS_show_track_base) {
      for (auto &kfp : keyFrames) {
        cv::Mat3b kim = kfp.second.drawDepthedFrame(minDepthCol, maxDepthCol);
        cv::imshow(
            "kf" + std::to_string(kfp.second.preKeyFrame->globalFrameNum), kim);
      }
      cv::Mat3b trackImg = baseKeyFrame().preKeyFrame->frameColored.clone();
      baseForTrack->draw(trackImg);
      cv::imshow("tracking base", trackImg);
      cv::waitKey();
      cv::destroyAllWindows();
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

void DsoSystem::printPointsInPly(std::ostream &out) {
  printInPly(out, pointHistory, pointHistoryCol);
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

} // namespace fishdso
