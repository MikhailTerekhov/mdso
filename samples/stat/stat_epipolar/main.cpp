#include "data/MultiFovReader.h"
#include "system/DelaunayDsoInitializer.h"
#include "system/DsoSystem.h"
#include "system/StereoMatcher.h"
#include "util/defs.h"
#include "util/flags.h"
#include "util/geometry.h"
#include "util/settings.h"
#include <atomic>
#include <ceres/solver.h>
#include <gflags/gflags.h>
#include <iostream>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

using namespace mdso;

const int totalFrames = 2500;

DEFINE_bool(show_epipolar, false,
            "Do we need to show epipolar curves searched?");

KeyFrame kfFromEpipolar(CameraModel *cam, const cv::Mat &baseFrame,
                        const cv::Mat &frameToTraceOn, const SE3 &firstToSecond,
                        PixelSelector &pixelSelector,
                        const Settings &settings) {
  KeyFrame result(cam, baseFrame, 0, pixelSelector, settings.keyFrame,
                  settings.pointTracer, settings.intensity,
                  settings.residualPattern, settings.pyramid);
  PreKeyFrame toTraceOn(cam, frameToTraceOn, 1, settings.pyramid);
  toTraceOn.worldToThis = firstToSecond;

  auto deb = FLAGS_show_epipolar ? ImmaturePoint::DRAW_EPIPOLE
                                 : ImmaturePoint::NO_DEBUG;
  for (const auto &ip : result.immaturePoints)
    ip->traceOn(toTraceOn, deb);

  return result;
}

DEFINE_int32(start_frame, 2, "First baseframe used.");
DEFINE_int32(end_frame, 2450, "Last baseframe used.");
DEFINE_int32(disparity_shift, 3,
             "Number of frames to skip between the tested ones when collecting "
             "disparity errors.");
DEFINE_double(disparity_trans_error, 0.012,
              "Translational error on one meter of displacement.");
DEFINE_double(disparity_rot_error, 0.003,
              "Rotational error on one meter of displacement.");
DEFINE_bool(precise_placement, false, "Use ground truth relative position.");
DEFINE_string(disparity_output, "disp_err.txt", "Disparity errors output file");

DEFINE_bool(show_all, false,
            "Do we need to show intermediate debug info (namely, keyframe with "
            "traced points on it)?");

DEFINE_bool(run_parallel, true, "Collect all in parallel?");

const int lastFrameNumGlobal = 2500;

struct EpiErr {
  double disparity;
  double expectedErr;
  double realErrBef;
  double realErr;
  double depthGT;
  double depth;
  double depthBeforeSubpixel;
  double eBeforeSubpixel, eAfterSubpixel;

  friend std::ostream &operator<<(std::ostream &os, const EpiErr &epiErr) {
    return os << epiErr.disparity << ' ' << epiErr.expectedErr << ' '
              << epiErr.realErrBef << ' ' << epiErr.realErr << ' '
              << epiErr.depthGT << ' ' << epiErr.depth << ' '
              << epiErr.depthBeforeSubpixel << ' ' << epiErr.eBeforeSubpixel
              << ' ' << epiErr.eAfterSubpixel;
  }
};

class CollectDisparities {
private:
  int *startFrames;
  const MultiFovReader *reader;
  std::vector<EpiErr> *errors;
  Settings settings;

public:
  CollectDisparities(const MultiFovReader *reader, int *startFrames,
                     std::vector<EpiErr> *errors, const Settings &settings)
      : startFrames(startFrames)
      , reader(reader)
      , errors(errors)
      , settings(settings) {}

  void operator()(const tbb::blocked_range<int> &range) const {
    PixelSelector pixelSelector;
    std::mt19937 mt;

    for (int ind = range.begin(); ind < range.end(); ++ind) {
      int firstFrameNum = startFrames[ind];
      int secondFrameNum = firstFrameNum + FLAGS_disparity_shift;
      auto firstToWorld = reader->frameToWorld(firstFrameNum);
      auto secondToWorld = reader->frameToWorld(secondFrameNum);
      CHEK(firstToWorld);
      CHECK(secondToWorld);
      SE3 firstToSecondGT =
          secondToWorld.value().inverse() * firstToWorld.value();
      SE3 firstToSecond;
      if (FLAGS_precise_placement)
        firstToSecond = firstToSecondGT;
      else {
        double dispDev = FLAGS_disparity_trans_error *
                         firstToSecondGT.translation().norm() / std::sqrt(3);
        double rotDev = M_PI / 180.0 * FLAGS_disparity_rot_error *
                        firstToSecondGT.translation().norm() / std::sqrt(3);

        std::normal_distribution<double> rotd(0, rotDev);
        std::normal_distribution<double> transd(0, dispDev);

        SO3 rot = SO3::exp(Vec3(rotd(mt), rotd(mt), rotd(mt))) *
                  firstToSecondGT.so3();
        Vec3 trans = firstToSecondGT.translation();
        for (int i = 0; i < 3; ++i)
          trans[i] += transd(mt);
        firstToSecond = SE3(rot, trans);
      }

      KeyFrame keyFrame =
          kfFromEpipolar(reader->cam.get(), reader->getFrame(firstFrameNum),
                         reader->getFrame(secondFrameNum), firstToSecond,
                         pixelSelector, settings);

      if (FLAGS_show_all) {
        std::vector<double> depths;
        for (const auto &ip : keyFrame.immaturePoints)
          if (ip->state == ImmaturePoint::ACTIVE && ip->maxDepth != INF)
            depths.push_back(ip->depth);
        setDepthColBounds(depths);
        cv::imshow("traced points",
                   keyFrame.drawDepthedFrame(minDepthCol, maxDepthCol));
        cv::waitKey();
      }

      cv::Mat1d dGT = reader->getDepths(firstFrameNum);
      for (const auto &ip : keyFrame.immaturePoints) {
        if (ip->state != ImmaturePoint::ACTIVE || ip->maxDepth == INF)
          continue;
        double depthGT = dGT(toCvPoint(ip->p));
        Vec2 reprojGT = reader->cam->map(
            firstToSecondGT *
            (depthGT * reader->cam->unmap(ip->p).normalized()));
        Vec2 reproj = reader->cam->map(
            firstToSecond *
            (ip->depth * reader->cam->unmap(ip->p).normalized()));
        Vec2 reprojBef = reader->cam->map(
            firstToSecond *
            (ip->depthBeforeSubpixel * reader->cam->unmap(ip->p).normalized()));

        Vec2 reprojInfD = reader->cam->map(
            firstToSecond.so3() * reader->cam->unmap(ip->p).normalized());
        EpiErr e;
        e.disparity = (reproj - reprojInfD).norm();
        e.expectedErr = std::sqrt(ip->lastFullVar);
        e.realErrBef = (reprojGT - reprojBef).norm();
        e.realErr = (reprojGT - reproj).norm();
        e.depthGT = depthGT;
        e.depth = ip->depth;
        e.depthBeforeSubpixel = ip->depthBeforeSubpixel;
        e.eBeforeSubpixel = ip->eBeforeSubpixel;
        e.eAfterSubpixel = ip->eAfterSubpixel;
        errors[ind].push_back(e);
      }
    }
  }
};

void collectEpipolarStat(const MultiFovReader &reader) {
  Settings settings = getFlaggedSettings();

  if (FLAGS_run_parallel)
    settings.threading.numThreads = 1;

  std::cout << "Start collecting dispaity errors..." << std::endl;
  const int lastFrame =
      std::min(FLAGS_end_frame, totalFrames - FLAGS_disparity_shift);
  const int testCount = lastFrame - FLAGS_start_frame + 1;
  int startFrames[testCount];
  std::vector<EpiErr> errors[testCount];
  for (int i = 0; i < testCount; ++i)
    startFrames[i] = FLAGS_start_frame + i;

  if (FLAGS_run_parallel) {
    tbb::parallel_for(
        tbb::blocked_range<int>(0, testCount),
        CollectDisparities(&reader, startFrames, errors, settings));
  } else {
    CollectDisparities(&reader, startFrames, errors,
                       settings)(tbb::blocked_range<int>(0, testCount));
  }

  std::vector<EpiErr> allErr;
  for (int it = 0; it < testCount; ++it)
    for (const auto &e : errors[it])
      allErr.push_back(e);

  outputArray(FLAGS_disparity_output, allErr);

  std::cout << "All done." << std::endl;
}

int main(int argc, char **argv) {
  std::string usage =
      R"abacaba(Usage: stat_epipolar data_dir
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
  collectEpipolarStat(reader);

  return 0;
}
