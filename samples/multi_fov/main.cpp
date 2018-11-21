#include "MultiFovReader.h"
#include "system/DsoSystem.h"
#include "system/StereoMatcher.h"
#include "util/settings.h"
#include <gflags/gflags.h>
#include <iostream>

using namespace fishdso;

SE3 predictBaseToThisDummy(const SE3 &baseToLbo, const SE3 &baseToLast) {
  return (baseToLast * baseToLbo.inverse()) * baseToLast;
}

void runTracker(const MultiFovReader &reader, int startFrameNum,
                int framesCount, double depthsNoiseLevel,
                StdVector<SE3> &worldToTracked,
                std::vector<AffineLightTransform<double>> &trackedAffLights) {
  StdVector<CameraModel> camPyr = reader.cam->camPyr();

  KeyFrame startFrame(reader.cam.get(), reader.getFrame(startFrameNum),
                      startFrameNum);
  startFrame.activateAllImmature();
  cv::Mat1f depths = reader.getDepths(startFrameNum);
  StdVector<Vec2> pnts;
  std::vector<cv::Point> cvPnts;
  std::vector<double> weights;
  std::vector<double> depthsVec;

  std::mt19937 mt;
  std::normal_distribution<double> depthsErrDistr(1, depthsNoiseLevel);

  for (const auto &ip : startFrame.immaturePoints) {
    double gtDepth = depths(toCvPoint(ip->p));
    double usedDepth = std::max(1e-3, gtDepth * depthsErrDistr(mt));
    // double usedDepth = gtDepth;

    pnts.push_back(ip->p);
    cvPnts.push_back(toCvPoint(ip->p));
    weights.push_back(1.0);
    depthsVec.push_back(usedDepth);
  }

  // setDepthColBounds(depthsVec);
  // cv::Mat depthedFrame = startFrame.frameColored.clone();
  // insertDepths(depthedFrame, pnts, depthsVec, minDepth, maxDepth, false);
  // cv::imshow("depths used", depthedFrame);
  // cv::waitKey();

  // for (int i = settingPyrLevels - 1; i >= 0; --i) {
  // cv::Mat pimgSm =
  // startFrame.preKeyFrame->drawDepthedFrame(i, minDepth, maxDepth);
  // cv::Mat pimg;
  // cv::resize(pimgSm, pimg,
  // cv::Size(reader.cam->getWidth(), reader.cam->getHeight()), 0,
  // 0, cv::INTER_NEAREST);
  // cv::imshow("pyr lvl #" + std::to_string(i), pimg);
  // cv::waitKey();
  // }

  FrameTracker tracker(
      camPyr, std::make_unique<DepthedImagePyramid>(
                  startFrame.preKeyFrame->frame(), cvPnts, depthsVec, weights));
  AffineLightTransform<double> affLight;
  SE3 baseToLbo;
  SE3 baseToLast = reader.getWorldToFrameGT(startFrameNum) *
                   reader.getWorldToFrameGT(startFrameNum - 1).inverse();

  worldToTracked.reserve(framesCount);
  worldToTracked.resize(0);
  worldToTracked.push_back(SE3());
  trackedAffLights.resize(framesCount);
  trackedAffLights.resize(0);
  trackedAffLights.push_back(AffineLightTransform<double>());

  for (int trackedNum = startFrameNum + 1;
       trackedNum <= startFrameNum + framesCount; ++trackedNum) {
    PreKeyFrame trackedFrame(reader.cam.get(), reader.getFrame(trackedNum),
                             trackedNum);
    AffineLightTransform<double> newAffLight;
    SE3 newBaseToLast;
    SE3 newBaseToLastPred = predictBaseToThisDummy(baseToLbo, baseToLast);
    std::tie(newBaseToLast, newAffLight) = tracker.trackFrame(
        ImagePyramid(trackedFrame.frame()), newBaseToLastPred, affLight);
    affLight = newAffLight;
    baseToLbo = baseToLast;
    baseToLast = newBaseToLast;
    worldToTracked.push_back(baseToLast);
    trackedAffLights.push_back(affLight);
  }

  SE3 firstToTrackedGT = reader.getWorldToFrameGT(startFrameNum + framesCount) *
                         reader.getWorldToFrameGT(startFrameNum).inverse();
  double transErr =
      180. / M_PI *
      angle(baseToLast.translation(), firstToTrackedGT.translation());
  double rotErr =
      180. / M_PI *
      (baseToLast.so3() * firstToTrackedGT.so3().inverse()).log().norm();

  LOG(INFO) << "trans, rot errors = " << transErr << ' ' << rotErr << std::endl;
  std::cout << "trans, rot errors = " << transErr << ' ' << rotErr << std::endl;
}

void compareReprojThresholds(const MultiFovReader &reader) {
  double values[] = {4., 3., 2., 1.75, 1.5, 1.25, 1., 0.75, 0.5, 0.25};
  const int testCount = 10;
  int startFrames[testCount] = {1,    410, 530,  570,  650,
                                1000, 810, 1170, 1115, 1400};

  int it = 0;

  std::ofstream tbl("errors.csv");
  tbl << "theshold, avg translational angle err, avg rotational angle err"
      << std::endl;

  StereoMatcher matcher(reader.cam.get());
  for (double val : values) {
    const int frameShift = 15;
    double sumTransErr = 0;
    double sumRotErr = 0;
    settingEssentialReprojErrThreshold = val;
    for (int firstFrame : startFrames) {
      LOG(INFO) << it << ") val = " << val << " firstFrame = " << firstFrame
                << std::endl;
      int secondFrame = firstFrame + frameShift;
      StdVector<Vec2> pnts[2];
      std::vector<double> depths[2];
      cv::Mat frames[2];
      frames[0] = reader.getFrame(firstFrame);
      frames[1] = reader.getFrame(secondFrame);
      SE3 estim = matcher.match(frames, pnts, depths);
      SE3 gt = reader.getWorldToFrameGT(secondFrame) *
               reader.getWorldToFrameGT(firstFrame).inverse();
      double transErr =
          180. / M_PI * angle(estim.translation(), gt.translation());
      double rotErr =
          180. / M_PI * (estim.so3() * gt.so3().inverse()).log().norm();
      LOG(INFO) << "trans, rot errors = " << transErr << ' ' << rotErr
                << std::endl;
      sumTransErr += transErr;
      sumRotErr += rotErr;

      std::cout << it << ") trans, rot errors = " << transErr << ' ' << rotErr
                << std::endl;

      // setDepthColBounds(depths[0]);
      // cv::Mat depthed0 = frames[0].clone();
      // insertDepths(depthed0, pnts[0], depths[0], minDepth, maxDepth, true);
      // cv::Mat depthed1 = frames[1].clone();
      // insertDepths(depthed1, pnts[1], depths[1], minDepth, maxDepth, true);
      // cv::imshow("frame 0", depthed0);
      // cv::imshow("frame 1", depthed1);
      // cv::waitKey();

      ++it;
    }

    double avgTransErr = sumTransErr / testCount;
    double avgRotErr = sumRotErr / testCount;

    LOG(INFO) << "for thres = " << val << " avg trans and rot errors are "
              << avgTransErr << ' ' << avgRotErr << std::endl;
    std::cout << "for thres = " << val << " avg trans and rot errors are "
              << avgTransErr << ' ' << avgRotErr << std::endl;
    tbl << val << ", " << avgTransErr << ", " << avgRotErr << std::endl;
  }

  tbl.close();
}

void compareDirectThresholds(const MultiFovReader &reader) {
  int valCount = 19;
  double values[] = {3,     7,  8,     9,    9.5,   10, 10.5, 11, 11.25, 11.5,
                     11.75, 12, 12.25, 12.5, 12.75, 13, 13.5, 14, 15};

  const int framesCount = 2500;
  const int trackCount = 10;
  const int candCount = 500;
  const int testCount = 20;
  const double depthsNoizeLevel = 0.05;
  std::mt19937 mt;
  std::uniform_int_distribution<> startDistr(2, framesCount - trackCount);
  int starts[testCount];

  std::vector<std::pair<int, double>> places;
  for (int st = 2; st < framesCount - trackCount - 2; ++st) {
    SE3 nextMotion = reader.getWorldToFrameGT(st + 1) *
                     reader.getWorldToFrameGT(st).inverse();
    places.push_back({st, nextMotion.log().norm()});
  }
  std::sort(places.begin(), places.end(),
            [](auto a, auto b) { return a.second > b.second; });
  for (int i = 0; i < testCount; ++i)
    starts[i] = places[i].first;

  StdVector<CameraModel> camPyr = reader.cam->camPyr();

  std::ofstream tbl("direct_err4.csv");
  tbl << "reproj thres, avg trans err, avg rot err" << std::endl;

  int it = 0;
  for (double val : values) {
    settingTrackingOutlierIntensityDiff = val;

    double sumTransErr = 0;
    double sumRotErr = 0;

    double sumTransErrPred = 0;
    double sumRotErrPred = 0;

    for (int st : starts) {
      LOG(INFO) << it << ") val = " << val << " firstFrame = " << st
                << std::endl;
      KeyFrame startFrame(reader.cam.get(), reader.getFrame(st), st);
      cv::Mat1f depths = reader.getDepths(st);
      StdVector<Vec2> pnts;

      // minDepth = 4;
      // maxDepth = 40;
      // cv::Mat3b dimg(reader.cam->getHeight(), reader.cam->getWidth());
      // for (int y = 0; y < dimg.rows; ++y)
      // for (int x = 0; x < dimg.cols; ++x)
      // dimg(y, x) = toCvVec3bDummy(depthCol(depths(y, x), minDepth,
      // maxDepth));
      // cv::imshow("depths!", dimg);
      // cv::waitKey();
      // exit(0);

      StdVector<SE3> worldToTracked;
      std::vector<AffineLightTransform<double>> trackedAffLights;
      runTracker(reader, st, trackCount, depthsNoizeLevel, worldToTracked,
                 trackedAffLights);

      SE3 baseToLast = worldToTracked.back();
      SE3 firstToTrackedGT = reader.getWorldToFrameGT(st + trackCount) *
                             reader.getWorldToFrameGT(st).inverse();
      double transErr =
          180. / M_PI *
          angle(baseToLast.translation(), firstToTrackedGT.translation());
      double rotErr =
          180. / M_PI *
          (baseToLast.so3() * firstToTrackedGT.so3().inverse()).log().norm();
      sumTransErr += transErr;
      sumRotErr += rotErr;

      LOG(INFO) << "trans, rot errors = " << transErr << ' ' << rotErr
                << std::endl;
      std::cout << it << ") trans, rot errors = " << transErr << ' ' << rotErr
                << std::endl;
      // std::cout << it << ") predicted  errors = " << transErrPred << ' '
      // << rotErrPred << std::endl;
      ++it;
      std::cout << 100. * double(it) / (valCount * testCount)
                << "% of cases processed\n";
    }

    double avgTransErr = sumTransErr / testCount;
    double avgRotErr = sumRotErr / testCount;

    LOG(INFO) << "for thres = " << val << " avg trans and rot errors are "
              << avgTransErr << ' ' << avgRotErr << std::endl;
    std::cout << "for thres = " << val << " avg trans and rot errors are "
              << avgTransErr << ' ' << avgRotErr << std::endl;
    tbl << val << ", " << avgTransErr << ", " << avgRotErr << std::endl;
  }

  tbl.close();
}

DEFINE_double(depths_noize, 0.05,
              "Multiplicative Gaussian noize with this standart deviation is "
              "applied to the base keyframe depths.");

DEFINE_int32(start_frame, 2, "Frame to start tracking from.");
DEFINE_int32(track_count, 50, "Number of frames to track.");

int main(int argc, char **argv) {
  std::string usage =
      R"abacaba(Usage: multi_fov data_dir
Where data_dir names a directory with MultiFoV fishseye dataset.
It should contain "info" and "data" subdirectories.)abacaba";

  gflags::ParseCommandLineFlags(&argc, &argv, true);
  gflags::SetUsageMessage(usage);
  google::InitGoogleLogging(argv[0]);

  if (argc != 2) {
    std::cerr << "Wrong number of arguments!\n" << usage << std::endl;
    return 1;
  }

  MultiFovReader reader(argv[1]);

  StdVector<SE3> worldToTracked;
  std::vector<AffineLightTransform<double>> trackedAffLights;
  runTracker(reader, FLAGS_start_frame, FLAGS_track_count, FLAGS_depths_noize,
             worldToTracked, trackedAffLights);
  StdVector<SE3> worldToTrackedGT;
  worldToTrackedGT.reserve(FLAGS_track_count);
  SE3 worldToFirst = reader.getWorldToFrameGT(FLAGS_start_frame);
  for (int frameNum = FLAGS_start_frame + 1;
       frameNum <= FLAGS_start_frame + FLAGS_track_count; ++frameNum)
    worldToTrackedGT.push_back(reader.getWorldToFrameGT(frameNum) *
                               worldToFirst.inverse());

  std::ofstream trackedOfs(FLAGS_output_directory + "/tracked_pos.txt");
  std::ofstream gtOfs(FLAGS_output_directory + "/ground_truth_pos.txt");

  int i = 1;
  for (const SE3 &motion : worldToTracked) {
    trackedOfs << i++ << ' ';
    putMotion(trackedOfs, motion);
    trackedOfs << std::endl;
  }

  i = 1;
  for (const SE3 &motion : worldToTrackedGT) {
    gtOfs << i++ << ' ';
    putMotion(gtOfs, motion);
    gtOfs << std::endl;
  }

  return 0;
}
