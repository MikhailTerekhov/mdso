#include "MultiFovReader.h"
#include "system/DelaunayDsoInitializer.h"
#include "system/DsoSystem.h"
#include "system/StereoMatcher.h"
#include "util/defs.h"
#include "util/geometry.h"
#include "util/settings.h"
#include <atomic>
#include <ceres/solver.h>
#include <gflags/gflags.h>
#include <iostream>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

using namespace fishdso;

const int depthErrCount = 12;
SE3 predictBaseToThisDummy(const SE3 &baseToLbo, const SE3 &baseToLast) {
  return (baseToLast * baseToLbo.inverse()) * baseToLast;
}

void runTracker(const MultiFovReader &reader, int startFrameNum,
                int framesCount, double depthsNoiseLevel,
                StdVector<SE3> &worldToTracked,
                std::vector<AffineLightTransform<double>> &trackedAffLights) {
  StdVector<CameraModel> camPyr = reader.cam->camPyr();
  PixelSelector pixelSelector;

  KeyFrame startFrame(reader.cam.get(), reader.getFrame(startFrameNum),
                      startFrameNum, pixelSelector);
  startFrame.activateAllImmature();
  cv::Mat1f depths = reader.getDepths(startFrameNum);
  StdVector<Vec2> pnts;
  std::vector<cv::Point> cvPnts;
  std::vector<double> weights;
  std::vector<double> depthsVec;

  std::mt19937 mt;
  std::normal_distribution<double> depthsErrDistr(1, depthsNoiseLevel);

  for (const auto &op : startFrame.optimizedPoints) {
    double gtDepth = depths(toCvPoint(op->p));
    double usedDepth = std::max(1e-3, gtDepth * depthsErrDistr(mt));
    // double usedDepth = gtDepth;

    pnts.push_back(op->p);
    cvPnts.push_back(toCvPoint(op->p));
    weights.push_back(1.0);
    depthsVec.push_back(usedDepth);
  }

  // setDepthColBounds(depthsVec);
  // cv::Mat depthedFrame = startFrame.preKeyFrame->frameColored.clone();
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

std::array<KeyFrame, 2> kfFromEpipolar(CameraModel *cam, const cv::Mat &frame1,
                                       const cv::Mat &frame2,
                                       const SE3 &fToSUsed,
                                       PixelSelector &pixelSelector) {
  std::array<KeyFrame, 2> result{KeyFrame(cam, frame1, 0, pixelSelector),
                                 KeyFrame(cam, frame2, 1, pixelSelector)};
  result[1].preKeyFrame->worldToThis = fToSUsed;

  for (int kfInd : {0, 1}) {
    int ipInd = 0;
    int intPrc = 0;
    for (const auto &ip : result[kfInd].immaturePoints) {
      ip->traceOn(*result[(kfInd + 1) % 2].preKeyFrame,
                  ImmaturePoint::NO_DEBUG);
    }
  }

  return result;
}

void testEpipolar(const MultiFovReader &reader, int fnum1, int fnum2) {
  std::cout << "start" << std::endl;

  PixelSelector pixelSelector;

  SE3 firstToSecondGT = reader.getWorldToFrameGT(fnum2) *
                        reader.getWorldToFrameGT(fnum1).inverse();
  cv::Mat frames[2] = {reader.getFrame(fnum1), reader.getFrame(fnum2)};
  StereoMatcher matcher(reader.cam.get());
  StdVector<Vec2> keyPoints[2];
  std::vector<double> depths[2];
  SE3 firstToSecond = matcher.match(frames, keyPoints, depths);

  setDepthColBounds(depths[1]);

  auto kf = kfFromEpipolar(reader.cam.get(), frames[0], frames[1],
                           firstToSecondGT, pixelSelector);

  double scale = firstToSecondGT.translation().norm() /
                 kf[1].preKeyFrame->worldToThis.translation().norm();
  std::vector<double> relErr;
  for (int kfNum = 0; kfNum < 1; ++kfNum) {
    cv::Mat1f dGT = reader.getDepths(kfNum == 0 ? fnum1 : fnum2);
    for (const auto &ip : kf[kfNum].immaturePoints) {
      double curGT = dGT(toCvPoint(ip->p));
      relErr.push_back(std::abs(scale * ip->depth - curGT) / curGT);
    }
  }
  std::sort(relErr.begin(), relErr.end());
  std::ofstream es("relerr.txt");
  for (double e : relErr)
    es << e << ' ';
  es << std::endl;
  es.close();

  cv::imshow("from epipolar", kf[1].drawDepthedFrame(minDepthCol, maxDepthCol));
  cv::waitKey();
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

  PixelSelector pixelSelector;

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
      KeyFrame startFrame(reader.cam.get(), reader.getFrame(st), st,
                          pixelSelector);
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
DEFINE_int32(min_stereo_displ, 20,
             "Minimum number of frames to skip between stereo frames.");
DEFINE_int32(max_stereo_displ, 20,
             "Maximum number of frames to skip between stereo frames.");
DEFINE_int32(max_DSO_input, 40,
             "Maximum number of frames to put into DSO when collecting errors "
             "on full DSO run.");
DEFINE_double(
    occlusion_thres, 0.1,
    "Relative translation error after which DSO is considered occluded");

DEFINE_bool(collect_tracking, true, "Do we need to collect tracking errors?");
DEFINE_bool(collect_stereo, true,
            "Do we need to collect stereo matching errors?");
DEFINE_bool(collect_BA, true,
            "Do we need to collect post-stereo optimizations errors?");
DEFINE_bool(collect_full_DSO, false,
            "Do we need to collect errors on full DSO run?");
DEFINE_int32(dso_test_count, 100,
             "Number of tests to collect full DSO errors from");
DEFINE_bool(
    show_all, false,
    "Do we need to show every step as an image when collecting full DSO?");
DEFINE_bool(collect_tracing, true, "Do we need to collect tracing errors?");

DEFINE_bool(collect_disparities, false,
            "Do we need to collect disparity errors?");
DEFINE_int32(disparity_shift, 15,
             "Number of frames to skip between the tested ones when collecting "
             "disparity errors.");
DEFINE_double(disparity_trans_error, 0.02,
              "Relative mean square deviation for translation  noize when "
              "collecing disparity errors.");
DEFINE_double(disparity_rot_error, 3,
              "Mean square deviation for rotation noize in degrees when "
              "collecing disparity errors.");
"

    DEFINE_bool(run_parallel, true, "Collect all in parallel?");

const int lastFrameNumGlobal = 2500;

struct EpiErr {
  double disparity;
  double expectedErr;
  double realErr;
  double depthGT;
  double depth;

  friend std::ostream &operator<<(std::ostream &os, const EpiErr &epiErr) {
    return os << epiErr.disparity << ' ' << epiErr.expectedErr << ' '
              << epiErr.realErr << ' ' << epiErr.depthGT << ' ' << epiErr.depth;
  }
};

class CollectDisparities {
private:
  const MultiFovReader *reader;
  std::vector<EpiErr> *errors;

public:
  CollectDisparities(const MultiFovReader *reader, std::vector<EpiErr> *errors)
      : reader(reader), errors(errors) {}

  void operator()(const tbb::blocked_range<int> &range) const {
    PixelSelector pixelSelector;
    std::mt19937 mt;
    std::normal_distribution<double> rotd(
        M_PI / 180.0 * FLAGS_disparity_rot_error / std::sqrt(3));
    std::normal_distribution<double> transd(1, FLAGS_disparity_trans_error);

    int end = range.end();
    end = std::min(end, lastFrameNumGlobal - FLAGS_disparity_shift + 2);
    for (int firstFrameNum = range.begin(); firstFrameNum < end;
         ++firstFrameNum) {
      int secondFrameNum = firstFrameNum + FLAGS_disparity_shift;
      SE3 firstToSecondGT = reader->getWorldToFrameGT(secondFrameNum) *
                            reader->getWorldToFrameGT(firstFrameNum).inverse();

      SO3 rot =
          SO3::exp(Vec3(rotd(mt), rotd(mt), rotd(mt))) * firstToSecondGT.so3();
      Vec3 trans = firstToSecondGT.translation();
      for (int i = 0; i < 3; ++i)
        trans[i] *= transd(mt);
      SE3 firstToSecond(rot, trans);

      auto kf = kfFromEpipolar(
          reader->cam.get(), reader->getFrame(firstFrameNum),
          reader->getFrame(secondFrameNum), firstToSecond, pixelSelector);

      for (int kfNum : {0, 1}) {
        SE3 curToOther = (kfNum == 0 ? firstToSecond : firstToSecond.inverse());
        SE3 curToOtherGT =
            (kfNum == 0 ? firstToSecondGT : firstToSecondGT.inverse());

        cv::Mat1f dGT =
            reader->getDepths(kfNum == 0 ? firstFrameNum : secondFrameNum);
        for (const auto &ip : kf[kfNum].immaturePoints) {
          if (ip->state != ImmaturePoint::ACTIVE || ip->maxDepth == INF)
            continue;
          double depthGT = dGT(toCvPoint(ip->p));
          Vec2 reprojGT = reader->cam->map(
              curToOtherGT * (depthGT * reader->cam->unmap(ip->p)));
          Vec2 reproj = reader->cam->map(
              curToOther * (ip->depth * reader->cam->unmap(ip->p)));
          Vec2 reprojInfD =
              reader->cam->map(curToOther.so3() * reader->cam->unmap(ip->p));
          EpiErr e;
          e.disparity = (reproj - reprojInfD).norm();
          e.expectedErr = std::sqrt(ip->lastFullVar);
          e.realErr = (reprojGT - reproj).norm();

          e.depthGT = depthGT;
          e.depth = ip->depth;
          errors[firstFrameNum].push_back(e);
        }
      }
    }
  }
};

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
            (baseToLast.translation() - firstToTrackedGT.translation()).norm();
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

  // transErrors and rotErrors are arrays of two MatXX, in which first ones
  // stand for errors before BA, and the second ones -- after
  MatXX *transErrors;
  MatXX *rotErrors;

  // depthErrors is an array of MatXX, in which
  // 0..3  occupy .25, .5, .75, .95 quantiles of relative errors in keypoints
  // 4..7  occupy the same quantiles after triangulation
  // 8..11 occupy the same quantiles after BA

  MatXX *depthErrors;

  bool performBA;

public:
  CollectStereo(const MultiFovReader *reader, const StereoMatcher *matcher,
                MatXX *transErrors, MatXX *rotErrors, MatXX *depthErrors,
                bool performBA)
      : reader(reader), matcher(matcher), transErrors(transErrors),
        rotErrors(rotErrors), depthErrors(depthErrors), performBA(performBA) {}

  void operator()(const tbb::blocked_range<int> &range) const {
    PixelSelector pixelSelector;

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
            (medianMult * estim.translation() - gt.translation()).norm();
        double rotErr =
            180. / M_PI * (estim.so3() * gt.so3().inverse()).log().norm();

        int row = baseFrameNum - FLAGS_start_frame;
        int col = refFrameNum - baseFrameNum - FLAGS_min_stereo_displ;

        transErrors[0](row, col) = transErr;
        rotErrors[0](row, col) = rotErr;

        depthErrors[0](row, col) = err25;
        depthErrors[1](row, col) = err50;
        depthErrors[2](row, col) = err75;
        depthErrors[3](row, col) = err95;

        if (performBA) {
          int frameNums[2] = {baseFrameNum, refFrameNum};
          std::vector<KeyFrame> keyFrames =
              DelaunayDsoInitializer::createKeyFramesDelaunay(
                  reader->cam.get(), frames, frameNums, pnts, depths, estim,
                  &pixelSelector, DelaunayDsoInitializer::NO_DEBUG);

          std::vector<double> errorsAfterTri;
          errorsAfterTri.reserve(keyFrames[0].optimizedPoints.size() +
                                 keyFrames[1].optimizedPoints.size());
          for (int it = 0; it < 2; ++it)
            for (const auto &op : keyFrames[it].optimizedPoints) {
              double depthGT = depthsGTImg[it](toCvPoint(op->p));
              errors.push_back(std::abs(medianMult * op->depth() - depthGT) /
                               depthGT);
            }
          std::sort(errors.begin(), errors.end());
          double err25Tri = errors[errors.size() * 0.25];
          double err50Tri = errors[errors.size() * 0.5];
          double err75Tri = errors[errors.size() * 0.75];
          double err95Tri = errors[errors.size() * 0.95];

          depthErrors[4](row, col) = err25Tri;
          depthErrors[5](row, col) = err50Tri;
          depthErrors[6](row, col) = err75Tri;
          depthErrors[7](row, col) = err95Tri;

          BundleAdjuster bundleAdjuster(reader->cam.get());
          for (int it = 0; it < 2; ++it)
            bundleAdjuster.addKeyFrame(&keyFrames[it]);
          bundleAdjuster.adjust(settingMaxFirstBAIterations);

          SE3 newEstim = keyFrames[1].preKeyFrame->worldToThis;
          double newTransErr =
              (medianMult * newEstim.translation() - gt.translation()).norm();
          double newRotErr =
              180. / M_PI * (newEstim.so3() * gt.so3().inverse()).log().norm();

          transErrors[1](row, col) = newTransErr;
          rotErrors[1](row, col) = newRotErr;

          for (int it = 0; it < 2; ++it)
            for (const auto &op : keyFrames[it].optimizedPoints) {
              double depthGT = depthsGTImg[it](toCvPoint(op->p));
              errors.push_back(std::abs(medianMult * op->depth() - depthGT) /
                               depthGT);
            }
          std::sort(errors.begin(), errors.end());
          double err25BA = errors[errors.size() * 0.25];
          double err50BA = errors[errors.size() * 0.5];
          double err75BA = errors[errors.size() * 0.75];
          double err95BA = errors[errors.size() * 0.95];

          depthErrors[8](row, col) = err25BA;
          depthErrors[9](row, col) = err50BA;
          depthErrors[10](row, col) = err75BA;
          depthErrors[11](row, col) = err95BA;
        }
      }
  }
};

std::atomic_int collectedCounter(0);

class CollectFullDSO {
private:
  const MultiFovReader *reader;
  int *startFrames;
  int *correctlyTracked;
  std::vector<double> *depthErrors;
  std::vector<double> *kpDepthErrors;
  std::vector<double> *epipolarDepthErrors;
  std::vector<EpiErr> *trackedDisp;

public:
  CollectFullDSO(const MultiFovReader *reader, int *startFrames,
                 int *correctlyTracked, std::vector<double> *depthErrors,
                 std::vector<double> *kpDepthErrors,
                 std::vector<double> *epipolarDepthErrors,
                 std::vector<EpiErr> *trackedDisp)
      : reader(reader), startFrames(startFrames),
        correctlyTracked(correctlyTracked), depthErrors(depthErrors),
        kpDepthErrors(kpDepthErrors), epipolarDepthErrors(epipolarDepthErrors),
        trackedDisp(trackedDisp) {}

  void operator()(const tbb::blocked_range<int> &range) const {
    PixelSelector pixelSelector;

    for (int ind = range.begin(); ind < range.end(); ++ind) {
      bool isInit = false;
      int initFrameNum = 0;
      DsoSystem dsoSystem(reader->cam.get());
      double scale = 0;
      SE3 worldToFirstGT = reader->getWorldToFrameGT(startFrames[ind]);
      int lastFrameNum = std::min(startFrames[ind] + FLAGS_max_DSO_input - 1,
                                  lastFrameNumGlobal);
      bool didOcclude = false;

      for (int curFrameNum = startFrames[ind]; curFrameNum <= lastFrameNum;
           ++curFrameNum) {
        SE3 firstToCurGT =
            reader->getWorldToFrameGT(curFrameNum) * worldToFirstGT.inverse();

        if (isInit) {
          SE3 forDSO(firstToCurGT.so3(), firstToCurGT.translation() / scale);
          dsoSystem.addGroundTruthPose(curFrameNum, forDSO);
        }

        std::shared_ptr<PreKeyFrame> nextFrame =
            dsoSystem.addFrame(reader->getFrame(curFrameNum), curFrameNum);

        if (!nextFrame)
          continue;

        if (!isInit) {
          isInit = true;
          initFrameNum = curFrameNum;

          SE3 firstToLastInitGT =
              reader->getWorldToFrameGT(
                  dsoSystem.lastInitialized->preKeyFrame->globalFrameNum) *
              worldToFirstGT.inverse();
          SE3 firstToLastInit =
              dsoSystem.lastInitialized->preKeyFrame->worldToThis;
          scale = firstToLastInitGT.translation().norm() /
                  firstToLastInit.translation().norm();
          LOG(INFO) << "scale = " << scale << std::endl;

          StdVector<Vec2> pnts;
          std::vector<double> depthsToDraw;

          cv::Mat1f depthsGT = reader->getDepths(curFrameNum);
          for (const auto &op : dsoSystem.lastInitialized->optimizedPoints) {
            double depth = op->depth();
            double depthGT = depthsGT(toCvPoint(op->p));
            depthErrors[ind].push_back(std::abs(scale * depth - depthGT) /
                                       depthGT);

            if (FLAGS_show_all) {
              pnts.push_back(op->p);
              depthsToDraw.push_back(depth);
            }
          }

          if (FLAGS_show_all) {
            setDepthColBounds(depthsToDraw);
            cv::Mat imgGT =
                dsoSystem.lastInitialized->preKeyFrame->frameColored.clone();
            insertDepths(imgGT, pnts, depthsToDraw, minDepthCol, maxDepthCol,
                         false);
            cv::imshow("interpolated depths", imgGT);
          }

          for (const auto &kp : dsoSystem.lastKeyPointDepths) {
            double depth = kp.second;
            double depthGT = depthsGT(toCvPoint(kp.first));
            kpDepthErrors[ind].push_back(std::abs(scale * depth - depthGT) /
                                         depthGT);
          }

          // collect epipolar
          auto kf = kfFromEpipolar(
              reader->cam.get(), reader->getFrame(startFrames[ind]),
              reader->getFrame(initFrameNum), firstToLastInit, pixelSelector);

          for (int kfNum = 0; kfNum < 1; ++kfNum) {
            SE3 curToOther =
                (kfNum == 0 ? firstToLastInit : firstToLastInit.inverse());
            SE3 curToOtherGT =
                (kfNum == 0 ? firstToLastInitGT : firstToLastInitGT.inverse());

            cv::Mat1f dGT =
                reader->getDepths(kfNum == 0 ? startFrames[ind] : initFrameNum);
            for (const auto &ip : kf[kfNum].immaturePoints) {
              if (ip->state != ImmaturePoint::ACTIVE || ip->maxDepth == INF)
                continue;
              double depthGT = dGT(toCvPoint(ip->p));
              epipolarDepthErrors[ind].push_back(
                  std::abs(scale * ip->depth - depthGT) / depthGT);

              if (FLAGS_collect_disparities) {
                Vec2 reprojGT = reader->cam->map(
                    curToOtherGT * (depthGT * reader->cam->unmap(ip->p)));
                Vec2 reproj = reader->cam->map(
                    curToOther * (ip->depth * reader->cam->unmap(ip->p)));
                Vec2 reprojInfD = reader->cam->map(curToOther.so3() *
                                                   reader->cam->unmap(ip->p));
                EpiErr e;
                e.disparity = (reproj - reprojInfD).norm();
                e.expectedErr = std::sqrt(ip->lastFullVar);
                e.realErr = (reprojGT - reproj).norm();
                e.depthGT = depthGT;
                e.depth = scale * ip->depth;
                trackedDisp[ind].push_back(e);
              }
            }
          }
          if (FLAGS_show_all) {
            cv::imshow("from epipolar",
                       kf[1].drawDepthedFrame(minDepthCol, maxDepthCol));
            cv::waitKey();
          }

          continue;
        }

        if (FLAGS_collect_tracking) {
          SE3 firstToCur = nextFrame->worldToThis;
          double err =
              ((scale * firstToCur.translation()) - firstToCurGT.translation())
                  .norm() /
              firstToCurGT.translation().norm();

          if (err > FLAGS_occlusion_thres) {
            correctlyTracked[ind] = curFrameNum - initFrameNum;
            didOcclude = true;
            break;
          }
        }
      }

      if (!didOcclude)
        correctlyTracked[ind] = lastFrameNum - initFrameNum;

      LOG(WARNING) << "complete count " << ++collectedCounter
                   << "(ind = " << ind << ", start frame = " << startFrames[ind]
                   << ")" << std::endl
                   << "corrtracked = " << correctlyTracked[ind] << std::endl;
    }
  }
};

void collectStatistics(const MultiFovReader &reader) {
  if (FLAGS_run_parallel)
    FLAGS_num_threads = 1;

  if (FLAGS_collect_tracking) {
    std::cout << "Collect tracking errors..." << std::endl;
    int lastFrameNum =
        std::min(FLAGS_end_frame, lastFrameNumGlobal - FLAGS_track_count);

    MatXX transErrors(lastFrameNum - FLAGS_start_frame + 1, FLAGS_track_count);
    MatXX rotErrors(lastFrameNum - FLAGS_start_frame + 1, FLAGS_track_count);
    if (FLAGS_run_parallel) {
      tbb::parallel_for(
          tbb::blocked_range<int>(FLAGS_start_frame, lastFrameNum + 1),
          CollectTracking(&reader, &transErrors, &rotErrors));
    } else {
      CollectTracking(&reader, &transErrors, &rotErrors)(
          tbb::blocked_range<int>(FLAGS_start_frame, lastFrameNum + 1));
    }

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
    std::cout << "Tracking errors collected." << std::endl;
  }

  if (FLAGS_collect_stereo) {
    std::cout << "Start collecting stereo-initialization errors..."
              << std::endl;

    StereoMatcher matcher(reader.cam.get());
    int lastFrameNum =
        std::min(FLAGS_end_frame, lastFrameNumGlobal - FLAGS_max_stereo_displ);
    int rows = lastFrameNum - FLAGS_start_frame + 1;
    int cols = FLAGS_max_stereo_displ - FLAGS_min_stereo_displ + 1;
    MatXX transErrors[2] = {MatXX(rows, cols), MatXX(rows, cols)};
    MatXX rotErrors[2] = {MatXX(rows, cols), MatXX(rows, cols)};
    MatXX depthErrors[depthErrCount];
    for (int i = 0; i < depthErrCount; ++i)
      depthErrors[i] = MatXX(rows, cols);

    if (FLAGS_run_parallel) {
      tbb::parallel_for(
          tbb::blocked_range<int>(FLAGS_start_frame, lastFrameNum + 1),
          CollectStereo(&reader, &matcher, transErrors, rotErrors, depthErrors,
                        FLAGS_collect_BA));
    } else {
      CollectStereo(&reader, &matcher, transErrors, rotErrors, depthErrors,
                    FLAGS_collect_BA)(
          tbb::blocked_range<int>(FLAGS_start_frame, lastFrameNum + 1));
    }

    std::ofstream stereoTable("stereo_err.csv");
    for (int baseFrameNum = FLAGS_start_frame; baseFrameNum <= lastFrameNum;
         baseFrameNum++)
      for (int refFrameNum = baseFrameNum + FLAGS_min_stereo_displ;
           refFrameNum <= baseFrameNum + FLAGS_max_stereo_displ;
           ++refFrameNum) {
        int row = baseFrameNum - FLAGS_start_frame;
        int col = refFrameNum - baseFrameNum - FLAGS_min_stereo_displ;
        stereoTable << baseFrameNum << "," << refFrameNum << ","
                    << transErrors[0](row, col) << "," << rotErrors[0](row, col)
                    << "," << transErrors[1](row, col) << ","
                    << rotErrors[1](row, col) << ",";
        for (int i = 0; i < depthErrCount - 1; ++i)
          stereoTable << depthErrors[i](row, col) << ",";
        stereoTable << depthErrors[depthErrCount - 1](row, col) << std::endl;
      }
    stereoTable.close();

    std::cout << "Stereo errors collected." << std::endl;
  }

  if (FLAGS_collect_full_DSO) {
    std::cout << "Start collecting full system run errors..." << std::endl;

    const int testCount = 9;
    // const int testCount = FLAGS_dso_test_count;
    // const int testCount = FLAGS_end_frame - FLAGS_start_frame + 1;
    int lastF = 2300;
    // int startFrames[testCount];
    int startFrames[testCount] = {410, 530,  570,  650, 1000,
                                  810, 1170, 1115, 1400};

    // std::vector<int> framesRot, framesStill;
    // framesRot.reserve(lastF);
    // framesStill.reserve(lastF);
    // for (int i = 0; i < lastF; ++i) {
    // SE3 mGT = reader.getWorldToFrameGT(i + FLAGS_first_frames_skip) *
    // reader.getWorldToFrameGT(i).inverse();
    // double angle = mGT.so3().log().norm();
    // if (angle > 10.0 * M_PI / 180.0)
    // framesRot.push_back(i);
    // else if (angle < 3.0 * M_PI / 180.0)
    // framesStill.push_back(i);
    // }
    // std::cout << "rot num = " << framesRot.size()
    // << "\nstillnum = " << framesStill.size() << std::endl;
    // std::mt19937 mt;
    // std::shuffle(framesRot.begin(), framesRot.end(), mt);
    // std::shuffle(framesStill.begin(), framesStill.end(), mt);

    // for (int i = 0; i < testCount / 2; ++i)
    // startFrames[i] = framesRot[i];
    // for (int i = 0; i < testCount / 2; ++i)
    // startFrames[i + testCount / 2] = framesStill[i];

    // for (int st = FLAGS_start_frame; st <= FLAGS_end_frame; ++st)
    // startFrames[st - FLAGS_start_frame] = st;

    int correctlyTracked[testCount];
    std::vector<double> depthErrors[testCount];
    std::vector<double> kpDepthErrors[testCount];
    std::vector<double> epipolarDepthErrors[testCount];
    std::vector<EpiErr> trackedDisp[testCount];
    int lastFrameNum =
        std::min(FLAGS_end_frame, lastFrameNumGlobal - FLAGS_max_DSO_input);

    FLAGS_continue_choosing_keyframes = false;
    FLAGS_write_files = false;

    if (FLAGS_run_parallel) {
      tbb::parallel_for(tbb::blocked_range<int>(0, testCount),
                        CollectFullDSO(&reader, startFrames, correctlyTracked,
                                       depthErrors, kpDepthErrors,
                                       epipolarDepthErrors, trackedDisp));
    } else {
      CollectFullDSO(&reader, startFrames, correctlyTracked, depthErrors,
                     kpDepthErrors, epipolarDepthErrors,
                     trackedDisp)(tbb::blocked_range<int>(0, testCount));
    }

    std::vector<double> allErrors;
    std::vector<double> allKpErrors;
    std::vector<double> allEpipolarErrors;
    std::vector<EpiErr> allDisp;

    for (int it = 0; it < testCount; ++it)
      for (double e : depthErrors[it])
        allErrors.push_back(e);
    for (int it = 0; it < testCount; ++it)
      for (double e : kpDepthErrors[it])
        allKpErrors.push_back(e);
    for (int it = 0; it < testCount; ++it)
      for (double e : epipolarDepthErrors[it])
        allEpipolarErrors.push_back(e);

    std::sort(allErrors.begin(), allErrors.end());
    std::sort(allKpErrors.begin(), allKpErrors.end());
    std::sort(allEpipolarErrors.begin(), allEpipolarErrors.end());

    for (int it = 0; it < testCount; ++it)
      for (const auto &e : trackedDisp[it])
        allDisp.push_back(e);

    outputArray("disp_err.txt", allDisp);

    outputArray("dso_depth_err.txt", allErrors);
    outputArray("dso_kp_depth_err.txt", allKpErrors);
    outputArray("dso_epi_depth_err.txt", allEpipolarErrors);
    outputArray("dso_tracked.txt", correctlyTracked, testCount);

    std::cout << "Full system errors collected." << std::endl;
  }

  std::cout << "All done." << std::endl;
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

  // visualizeTracking(argv[1], FLAGS_start_frame, FLAGS_track_count,
  // FLAGS_depths_noize);

  MultiFovReader reader(argv[1]);
  collectStatistics(reader);
  // testEpipolar(reader, 410, 420);

  return 0;
}
