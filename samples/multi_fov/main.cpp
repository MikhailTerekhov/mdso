#include "MultiFovReader.h"
#include "system/DsoSystem.h"
#include "system/StereoMatcher.h"
#include "util/settings.h"
#include <gflags/gflags.h>
#include <iostream>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

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
}

void visualizeTracking(const std::string &dataDir, int baseFrame,
                       int trackCount, double depthsNoize) {
  MultiFovReader reader(dataDir);

  StdVector<SE3> worldToTracked;
  std::vector<AffineLightTransform<double>> trackedAffLights;
  runTracker(reader, baseFrame, trackCount, depthsNoize, worldToTracked,
             trackedAffLights);
  StdVector<SE3> worldToTrackedGT;
  worldToTrackedGT.reserve(trackCount);
  SE3 worldToFirst = reader.getWorldToFrameGT(baseFrame);
  for (int frameNum = baseFrame + 1; frameNum <= baseFrame + trackCount;
       ++frameNum)
    worldToTrackedGT.push_back(reader.getWorldToFrameGT(frameNum) *
                               worldToFirst.inverse());

  std::ofstream trackedOfs("tracked_pos.txt");
  std::ofstream gtOfs("ground_truth_pos.txt");

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
              "Multiplicative Gaussian noize with this standard deviation is "
              "applied to the base keyframe depths.");

DEFINE_int32(start_frame, 2, "First baseframe used.");
DEFINE_int32(end_frame, 2450, "Last baseframe used.");
DEFINE_int32(track_count, 30,
             "Number of frames to track after each baseframe.");
DEFINE_int32(min_stereo_displ, 5,
             "Minimum number of frames to skip between stereo frames.");
DEFINE_int32(max_stereo_displ, 5,
             "Maximum number of frames to skip between stereo frames.");

DEFINE_bool(collect_tracking, true, "Do we need to collect tracking errors?");
DEFINE_bool(collect_stereo, true,
            "Do we need to collect stereo matching errors?");
DEFINE_bool(collect_BA, true,
            "Do we need to collect post-stereo optimizations errors?");

DEFINE_bool(run_parallel, true, "Collect all in parallel?");

const int lastFrameNumGlobal = 2500;

class CollectTracking {
private:
  const MultiFovReader *reader;
  MatXX *transErrors;
  MatXX *rotErrors;

public:
  CollectTracking(const MultiFovReader *reader, MatXX *transErrors,
                  MatXX *rotErrors)
      : reader(reader), transErrors(transErrors), rotErrors(rotErrors) {}

  void operator()(const tbb::blocked_range<int> &range) const {
    for (int baseFrameNum = range.begin(); baseFrameNum < range.end();
         ++baseFrameNum) {
      StdVector<SE3> worldToTracked;
      std::vector<AffineLightTransform<double>> affLightToTracked;
      runTracker(*reader, baseFrameNum, FLAGS_track_count, FLAGS_depths_noize,
                 worldToTracked, affLightToTracked);

      for (int trackedFrameNum = baseFrameNum + 1;
           trackedFrameNum <= baseFrameNum + FLAGS_track_count;
           ++trackedFrameNum) {
        SE3 baseToLast = worldToTracked.back();
        SE3 firstToTrackedGT =
            reader->getWorldToFrameGT(trackedFrameNum + FLAGS_track_count) *
            reader->getWorldToFrameGT(trackedFrameNum).inverse();
        double transErr =
            180. / M_PI *
            angle(baseToLast.translation(), firstToTrackedGT.translation());
        double rotErr =
            180. / M_PI *
            (baseToLast.so3() * firstToTrackedGT.so3().inverse()).log().norm();

        LOG(INFO) << "trans, rot errors = " << transErr << ' ' << rotErr
                  << std::endl;

        (*transErrors)(baseFrameNum - FLAGS_start_frame,
                       trackedFrameNum - baseFrameNum - 1) = transErr;
        (*rotErrors)(baseFrameNum - FLAGS_start_frame,
                     trackedFrameNum - baseFrameNum - 1) = rotErr;
      }
    }
  }
};

class CollectStereo {
private:
  const MultiFovReader *reader;
  const StereoMatcher *matcher;
  MatXX *transErrors;
  MatXX *rotErrors;
  MatXX *depthErr25;
  MatXX *depthErr50;
  MatXX *depthErr75;
  MatXX *depthErr95;

public:
  CollectStereo(const MultiFovReader *reader, const StereoMatcher *matcher,
                MatXX *transErrors, MatXX *rotErrors, MatXX *depthErr25,
                MatXX *depthErr50, MatXX *depthErr75, MatXX *depthErr95)
      : reader(reader), matcher(matcher), transErrors(transErrors),
        rotErrors(rotErrors), depthErr25(depthErr25), depthErr50(depthErr50),
        depthErr75(depthErr75), depthErr95(depthErr95) {}

  void operator()(const tbb::blocked_range<int> &range) const {
    for (int baseFrameNum = range.begin(); baseFrameNum < range.end();
         ++baseFrameNum)
      for (int refFrameNum = baseFrameNum + FLAGS_min_stereo_displ;
           refFrameNum <= baseFrameNum + FLAGS_max_stereo_displ;
           ++refFrameNum) {
        StdVector<Vec2> pnts[2];
        std::vector<double> depths[2];
        cv::Mat frames[2];
        frames[0] = reader->getFrame(baseFrameNum);
        frames[1] = reader->getFrame(refFrameNum);
        SE3 estim = matcher->match(frames, pnts, depths);
        SE3 gt = reader->getWorldToFrameGT(refFrameNum) *
                 reader->getWorldToFrameGT(baseFrameNum).inverse();

        std::vector<std::pair<int, double>> depthsGT[2];
        cv::Mat1f depthsGTImg[2] = {reader->getDepths(baseFrameNum),
                                    reader->getDepths(refFrameNum)};

        for (int it = 0; it < 2; ++it) {
          depthsGT[it].resize(depths[it].size());
          for (int i = 0; i < pnts[it].size(); ++i)
            depthsGT[it][i] = {i, depthsGTImg[it](toCvPoint(pnts[it][i]))};
          std::sort(depthsGT[it].begin(), depthsGT[it].end(),
                    [](auto a, auto b) { return a.second < b.second; });
        }

        auto median = depthsGT[0][depthsGT[0].size() / 2];
        double medianMult = median.second / depths[0][median.first];

        std::vector<double> errors;
        errors.reserve(depths[0].size() + depths[1].size());
        for (int it = 0; it < 2; ++it)
          for (int i = 0; i < pnts[it].size(); ++i)
            errors.push_back(
                std::abs(medianMult * depths[it][depthsGT[it][i].first] -
                         depthsGT[it][i].second) /
                depthsGT[it][i].second);

        std::sort(errors.begin(), errors.end());
        double err25 = errors[errors.size() * 0.25];
        double err50 = errors[errors.size() * 0.5];
        double err75 = errors[errors.size() * 0.75];
        double err95 = errors[errors.size() * 0.95];

        double transErr =
            180. / M_PI * angle(estim.translation(), gt.translation());
        double rotErr =
            180. / M_PI * (estim.so3() * gt.so3().inverse()).log().norm();

        int row = baseFrameNum - FLAGS_start_frame;
        int col = refFrameNum - baseFrameNum - FLAGS_min_stereo_displ;
        (*transErrors)(row, col) = transErr;
        (*rotErrors)(row, col) = rotErr;
        (*depthErr25)(row, col) = err25;
        (*depthErr50)(row, col) = err50;
        (*depthErr75)(row, col) = err75;
        (*depthErr95)(row, col) = err95;
      }
  }
};

void collectStatistics(const MultiFovReader &reader) {
  if (FLAGS_collect_tracking) {
    int lastFrameNum =
        std::min(FLAGS_end_frame, lastFrameNumGlobal - FLAGS_track_count);

    MatXX transErrors(lastFrameNum - FLAGS_start_frame + 1, FLAGS_track_count);
    MatXX rotErrors(lastFrameNum - FLAGS_start_frame + 1, FLAGS_track_count);
    tbb::parallel_for(
        tbb::blocked_range<int>(FLAGS_start_frame, lastFrameNum + 1),
        CollectTracking(&reader, &transErrors, &rotErrors));

    std::ofstream trackingTable("tracking_err.csv");
    for (int baseFrameNum = FLAGS_start_frame; baseFrameNum <= lastFrameNum;
         baseFrameNum++)
      for (int trackedFrameNum = baseFrameNum + 1;
           trackedFrameNum <= baseFrameNum + FLAGS_track_count;
           ++trackedFrameNum) {
        trackingTable << baseFrameNum << "," << trackedFrameNum << ","
                      << transErrors(baseFrameNum - FLAGS_start_frame,
                                     trackedFrameNum - baseFrameNum - 1)
                      << ","
                      << rotErrors(baseFrameNum - FLAGS_start_frame,
                                   trackedFrameNum - baseFrameNum - 1)
                      << std::endl;
      }
    trackingTable.close();
  }

  if (FLAGS_collect_stereo) {
    StereoMatcher matcher(reader.cam.get());
    int lastFrameNum =
        std::min(FLAGS_end_frame, lastFrameNumGlobal - FLAGS_max_stereo_displ);
    int rows = lastFrameNum - FLAGS_start_frame + 1;
    int cols = FLAGS_max_stereo_displ - FLAGS_min_stereo_displ + 1;
    MatXX transErrors(rows, cols);
    MatXX rotErrors(rows, cols);
    MatXX depthErr25(rows, cols);
    MatXX depthErr50(rows, cols);
    MatXX depthErr75(rows, cols);
    MatXX depthErr95(rows, cols);

    tbb::parallel_for(
        tbb::blocked_range<int>(FLAGS_start_frame, lastFrameNum + 1),
        CollectStereo(&reader, &matcher, &transErrors, &rotErrors, &depthErr25,
                      &depthErr50, &depthErr75, &depthErr95));
    std::ofstream stereoTable("stereo_err.csv");
    for (int baseFrameNum = FLAGS_start_frame; baseFrameNum <= lastFrameNum;
         baseFrameNum++)
      for (int refFrameNum = baseFrameNum + FLAGS_min_stereo_displ;
           refFrameNum <= baseFrameNum + FLAGS_max_stereo_displ;
           ++refFrameNum) {
        int row = baseFrameNum - FLAGS_start_frame;
        int col = refFrameNum - baseFrameNum - FLAGS_min_stereo_displ;
        stereoTable << baseFrameNum << "," << refFrameNum << ","
                    << transErrors(row, col) << "," << rotErrors(row, col)
                    << "," << depthErr25(row, col) << ","
                    << depthErr50(row, col) << "," << depthErr75(row, col)
                    << "," << depthErr95(row, col) << std::endl;
      }
    stereoTable.close();
  }
}

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
  collectStatistics(reader);

  return 0;
}
