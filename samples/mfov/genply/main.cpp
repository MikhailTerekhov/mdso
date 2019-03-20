#include "../reader/MultiFovReader.h"
#include "system/DsoSystem.h"
#include "util/defs.h"
#include "util/flags.h"
#include <iostream>

DEFINE_int32(start, 1, "Number of the starting frame.");
DEFINE_int32(count, 100, "Number of frames to process.");
DEFINE_int32(gt_points, 1'000'000,
             "Number of GT points in the generated cloud.");

DEFINE_bool(run_dso, true,
            "Do we need to run the system? If set to false, only GT pointcloud "
            "is generated (if gen_gt set to true).");

DEFINE_bool(gen_gt, true,
            "Do we need to generate GT pointcloud?");

int main(int argc, char **argv) {
  std::string usage =
      R"abacaba(Usage: genply data_dir
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

  int w = reader.cam->getWidth(), h = reader.cam->getHeight();
  int step =
      std::ceil(std::sqrt(double(FLAGS_count) * w * h / FLAGS_gt_points));
  std::cout << "step = " << step << std::endl;

  double scale = 2.0;
  if (FLAGS_run_dso) {
    std::vector<Vec3> points;
    std::vector<cv::Vec3b> colors;
    points.reserve(FLAGS_gt_points + FLAGS_count * 2000);
    colors.reserve(FLAGS_gt_points + FLAGS_count * 2000);

    Settings settings = getFlaggedSettings();
    settings.stereoMatcher.stereoGeometryEstimator.successProb = 0.999;
    // settings.pointTracer.positionVariance = 0.001;

    std::cout << "running DSO.." << std::endl;
    DsoSystem dso(reader.cam.get(), settings);
    for (int it = FLAGS_start; it < FLAGS_start + FLAGS_count; ++it) {
      std::cout << "add frame #" << it << std::endl;
      dso.addGroundTruthPose(it, reader.getWorldToFrameGT(it));
      dso.addFrame(reader.getFrame(it), it);
    }

    dso.fillRemainedHistory();
    scale = dso.scaleGTToOur;
  }

  if (FLAGS_gen_gt) {
    std::cout << "filling GT points..." << std::endl;
    // fill GT points
    std::vector<Vec3> pointsGT;
    std::vector<cv::Vec3b> colGT;
    SE3 baseGT = reader.getWorldToFrameGT(FLAGS_start);
    const double maxd = 1e10;
    for (int it = FLAGS_start; it < FLAGS_start + FLAGS_count; ++it) {
      SE3 curToBase = baseGT * reader.getWorldToFrameGT(it).inverse();
      cv::Mat1d depths = reader.getDepths(it);
      cv::Mat3b frame = reader.getFrame(it);
      for (int y = 0; y < h; y += step)
        for (int x = 0; x < w; x += step) {
          Vec3 p = reader.cam->unmap(Vec2(x, y));
          p.normalize();
          double d = depths(y, x);
          if (d > maxd)
            continue;
          p *= d;
          p = curToBase * p;
          p *= scale;
          pointsGT.push_back(p);
          colGT.push_back(frame(y, x));
        }
    }

    std::ofstream ofsGT(FLAGS_output_directory + "/pointsGT.ply");
    printInPly(ofsGT, pointsGT, colGT);
  }

  return 0;
}
