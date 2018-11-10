#include "system/DsoSystem.h"
#include "system/AffineLightTransform.h"
#include "system/StereoMatcher.h"
#include "util/settings.h"
#include <glog/logging.h>

namespace fishdso {

DsoSystem::DsoSystem(CameraModel *cam)
    : cam(cam), camPyr(cam->camPyr()), dsoInitializer(cam),
      isInitialized(false) {
  LOG(INFO) << "create DsoSystem" << std::endl;
}

DsoSystem::~DsoSystem() {
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
  worldToFrameGT[ind] = worldToThat;
}

void DsoSystem::addFrame(const cv::Mat &frame, int globalFrameNum) {
  LOG(INFO) << "add frame #" << globalFrameNum << std::endl;

  if (!isInitialized) {
    LOG(INFO) << "put into initializer" << std::endl;
    isInitialized = dsoInitializer.addFrame(frame, globalFrameNum);

    if (isInitialized) {
      LOG(INFO) << "initialization successful" << std::endl;
      std::vector<KeyFrame> kf =
          dsoInitializer.createKeyFrames(DsoInitializer::SPARSE_DEPTHS);
      for (const auto &f : kf)
        worldToFramePredict[f.preKeyFrame->globalFrameNum] =
            worldToFrame[f.preKeyFrame->globalFrameNum] =
                f.preKeyFrame->worldToThis;
      for (KeyFrame &keyFrame : kf)
        keyFrames.insert(std::pair<int, KeyFrame>(
            keyFrame.preKeyFrame->globalFrameNum, std::move(keyFrame)));

      bundleAdjuster = std::unique_ptr<BundleAdjuster>(new BundleAdjuster(cam));
      for (auto &p : keyFrames)
        bundleAdjuster->addKeyFrame(&p.second);

      std::ofstream ofsBeforeAdjust(FLAGS_output_directory +
                                    "/before_adjust.ply");
      printLastKfInPly(ofsBeforeAdjust);

      bundleAdjuster->adjust();

      if (FLAGS_switch_first_motion_to_GT) {
        SE3 worldToSecondKfGT =
            worldToFrameGT[keyFrames.rbegin()
                               ->second.preKeyFrame->globalFrameNum];
        keyFrames.rbegin()->second.preKeyFrame->worldToThis = worldToSecondKfGT;
        worldToFrame[keyFrames.rbegin()->second.preKeyFrame->globalFrameNum] =
            worldToSecondKfGT;
      }

      std::ofstream ofsAfterAdjust(FLAGS_output_directory +
                                   "/after_adjust.ply");
      printLastKfInPly(ofsAfterAdjust);

      frameTracker = std::unique_ptr<FrameTracker>(
          new FrameTracker(camPyr, baseKeyFrame().preKeyFrame.get()));
    }

    return;
  }

  std::unique_ptr<PreKeyFrame> preKeyFrame(
      new PreKeyFrame(frame, globalFrameNum));

  SE3 baseKfToCur;
  AffineLightTransform<double> lightBaseKfToCur;

  SE3 purePredicted = purePredictBaseKfToCur();
  SE3 predicted = predictBaseKfToCur();

  LOG(INFO) << "start tracking this frame" << std::endl;

  std::tie(baseKfToCur, lightBaseKfToCur) =
      frameTracker->trackFrame(preKeyFrame.get(), predicted, lightKfToLast);

  LOG(INFO) << "tracking ended" << std::endl;

  PreKeyFrame *baseKf = baseKeyFrame().preKeyFrame.get();
  preKeyFrame->lightWorldToThis = lightBaseKfToCur * baseKf->lightWorldToThis;

  worldToFrame[globalFrameNum] = baseKfToCur * baseKf->worldToThis;
  worldToFramePredict[globalFrameNum] = purePredicted * baseKf->worldToThis;
  preKeyFrame->worldToThis = worldToFrame[globalFrameNum];

  lightKfToLast = lightBaseKfToCur;

  LOG(INFO) << "estimated motion"
            << "\ntrans = " << baseKfToCur.translation()
            << "\nrot = " << baseKfToCur.unit_quaternion().coeffs().transpose()
            << std::endl;

  SE3 diff = baseKfToCur * predicted.inverse();
  LOG(INFO) << "diff to predicted:"
            << "\ntrans = " << diff.translation().norm()
            << "\nrot = " << diff.so3().log().norm() << std::endl;
  LOG(INFO) << "estimated aff = \n" << lightBaseKfToCur << std::endl;

  if (FLAGS_perform_tracking_check_GT) {
    std::cout << "perform comparison to ground truth" << std::endl;
    checkLastTrackedGT(std::move(preKeyFrame));
  } else if (FLAGS_perform_tracking_check_stereo) {
    std::cout << "perform stereo check" << std::endl;
    checkLastTrackedStereo(std::move(preKeyFrame));
  }
}

void DsoSystem::checkLastTrackedGT(std::unique_ptr<PreKeyFrame> lastFrame) {
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

  std::cout << "translation error distance = " << transErr
            << "\nrotation error angle = " << rotErr << std::endl;
}

void DsoSystem::checkLastTrackedStereo(std::unique_ptr<PreKeyFrame> lastFrame) {
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
  points.reserve(lastKeyFrame().interestPoints.size());
  for (const auto &ip : lastKeyFrame().interestPoints) {
    double d = ip.depthd();
    if (d != d || d > 1e3)
      continue;

    points.push_back({ip.p, d});
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
    cv::Vec3b color =
        lastKeyFrame().frameColored.at<cv::Vec3b>(toCvPoint(p.first));
    out << int(color[2]) << ' ' << int(color[1]) << ' ' << int(color[0])
        << std::endl;
  }
}

void putMotion(std::ostream &out, const SE3 &motion) {
  out << motion.unit_quaternion().coeffs().transpose() << ' ';
  out << motion.translation().transpose();
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
