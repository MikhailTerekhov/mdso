#include "data/RobotcarReader.h"
#include "util/PixelSelector.h"
#include "util/util.h"

DEFINE_string(chunk_dir, "/shared/datasets/oxford-robotcar/2015-08-21-16-23-57",
              "Name of the chunk to output pointcloud from.");
DEFINE_string(models_dir, "data/models/robotcar",
              "Directory with omnidirectional camera models. It is provided in "
              "our repository. IT IS NOT THE \"models\" DIRECTORY FROM THE "
              "ROBOTCAR DATASET SDK!");
DEFINE_string(extrinsics_dir, "thirdparty/robotcar-dataset-sdk/extrinsics",
              "Directory with RobotCar dataset sensor extrinsics, provided in "
              "the dataset SDK.");
DEFINE_string(rtk_dir, "/shared/datasets/oxford-robotcar/rtk",
              "Directory with RTK ground truth poses.");
DEFINE_bool(gen_clouds, true,
            "Do we need to generate whole clouds from the dump?");
DEFINE_string(
    out_front, "lidar_front.ply",
    "Output file name, to which the cloud from lms_front will be written "
    "in PLY format.");
DEFINE_string(
    out_rear, "lidar_rear.ply",
    "Output file name, to which the cloud from lms_rear will be written "
    "in PLY format.");
DEFINE_string(out_ldmrs, "lidar_ldmrs.ply",
              "Output file name, to which the cloud from ldmrs will be written "
              "in PLY format.");
DEFINE_string(out_traj_orig, "vo_orig.txt",
              "The file to write original VO trajectory into.");
DEFINE_string(out_traj_interp, "vo_interp.txt",
              "The file to write interpolated VO trajectory into.");
DEFINE_int32(interp_gran, 1000,
             "Number of poses to be output into interpolated vo trajectory");
DEFINE_int32(max_proj_points, 20'000, "Maximum number of projected points.");
DEFINE_int32(cloud_points, 20'000'000,
             "Number of points to be written into the cloud");
DEFINE_bool(fill_vo_gaps, false,
            "Do we need to fill the gaps in VO trajectory?");
DEFINE_string(
    out_project, "projected.png",
    "Name of the file to output the image with the projected cloud into.");
DEFINE_string(
    out_interp, "interpolated.png",
    "Name of the file to output the image with the interpolated depths.");
DEFINE_string(out_triang, "triangulations.png",
              "Name of the file to output the image with triangulations.");
DEFINE_int32(project_idx, 750,
             "Index of the frame to project point cloud onto.");
DEFINE_double(
    time_win, 5,
    "Lidar scans from the time window [ts - time_win, ts + time_win] will be "
    "used to draw the projected cloud. Here ts is the time, corresponding to "
    "project_idx frame. time_win is measured in seconds.");
DEFINE_double(
    rel_point_size, 0.001,
    "Relative to w+h point size on the images with projected points.");

cv::Mat3b project(const RobotcarReader &reader, int idx) {
  auto frame = reader.frame(idx);
  Timestamp baseTs = frame[0].timestamp;
  Timestamp timeWin = FLAGS_time_win * 1e6;
  Timestamp minTs = baseTs - timeWin, maxTs = baseTs + timeWin;
  std::vector<Vec3> lmsFrontCloud =
      reader.getLmsFrontCloud(minTs, maxTs, baseTs);
  std::vector<Vec3> lmsRearCloud = reader.getLmsRearCloud(minTs, maxTs, baseTs);
  std::vector<Vec3> ldmrsCloud = reader.getLdmrsCloud(minTs, maxTs, baseTs);
  std::vector<Vec3> cloud;
  cloud.reserve(lmsFrontCloud.size() + lmsRearCloud.size() + ldmrsCloud.size());
  cloud.insert(cloud.end(), lmsFrontCloud.begin(), lmsFrontCloud.end());
  cloud.insert(cloud.end(), lmsRearCloud.begin(), lmsRearCloud.end());
  cloud.insert(cloud.end(), ldmrsCloud.begin(), ldmrsCloud.end());
  cv::Mat3b images[RobotcarReader::numCams];

  int s = FLAGS_rel_point_size *
          (RobotcarReader::imageWidth + RobotcarReader::imageHeight) / 2;
  CameraBundle cam = reader.cam();
  for (int i = 0; i < RobotcarReader::numCams; ++i) {
    CHECK(frame[i].frame.channels() == 3);
    images[i] = frame[i].frame.clone();
    SE3 bodyToCam = cam.bundle[i].bodyToThis;
    StdVector<Vec2> points;
    std::vector<double> depths;
    for (const Vec3 &p : cloud) {
      Vec3 moved = bodyToCam * p;
      if (!reader.cam().bundle[i].cam.isMappable(moved))
        continue;

      double depth = moved.norm();
      Vec2 projected = reader.cam().bundle[i].cam.map(moved);
      depths.push_back(depth);
      points.push_back(projected);
    }

    if (i == 0)
      setDepthColBounds(depths);

    for (int j = 0; j < points.size(); ++j)
      putSquare(images[i], toCvPoint(points[j]), s,
                depthCol(depths[j], minDepthCol, maxDepthCol), cv::FILLED);
  }

  cv::Mat3b result;
  cv::hconcat(images, RobotcarReader::numCams, result);
  return result;
}

cv::Mat3b interpolate(const RobotcarReader &reader, int pointsNeeded, int idx) {
  auto frame = reader.frame(idx);
  CameraBundle cam = reader.cam();
  std::vector<PixelSelector> pixelSelectors(cam.bundle.size());
  std::vector<cv::Mat3b> interp(cam.bundle.size());
  auto depths = reader.depths(idx);
  cv::Mat3b framesDrawn[RobotcarReader::numCams];
  for (int ci = 0; ci < RobotcarReader::numCams; ++ci)
    framesDrawn[ci] = frame[ci].frame;
  cv::Mat3b triangs =
      dynamic_cast<RobotcarReader::Depths *>(depths.get())->draw(framesDrawn);
  cv::imwrite(FLAGS_out_triang, triangs);

  for (int ci = 0; ci < cam.bundle.size(); ++ci) {
    pixelSelectors[ci].initialize(frame[ci].frame,
                                  pointsNeeded / cam.bundle.size());
    cv::Mat1b frameGray = cvtBgrToGray(frame[ci].frame);
    cv::Mat1d gradX, gradY, gradNorm;
    grad(frameGray, gradX, gradY, gradNorm);
    PixelSelector::PointVector points = pixelSelectors[ci].select(
        frame[ci].frame, gradNorm, pointsNeeded / cam.bundle.size());
    std::vector<double> depthVals;
    std::vector<cv::Point> pointsSet;
    for (cv::Point p : points) {
      std::optional<double> d = depths->depth(ci, toVec2(p));
      if (d) {
        depthVals.push_back(d.value());
        pointsSet.push_back(p);
      }
    }

    interp[ci] = frame[ci].frame.clone();
    for (int i = 0; i < pointsSet.size(); ++i)
      cv::circle(interp[ci], pointsSet[i], 5,
                 depthCol(depthVals[i], minDepthCol, maxDepthCol), cv::FILLED);
  }

  cv::Mat3b result;
  cv::hconcat(interp.data(), interp.size(), result);
  return result;
}

void genClouds(const RobotcarReader &reader, Timestamp minTs, Timestamp maxTs,
               Timestamp baseTs) {
  constexpr int totalClouds = 3;
  std::vector<Vec3> lmsFrontCloud =
      reader.getLmsFrontCloud(minTs, maxTs, baseTs);
  std::vector<Vec3> lmsRearCloud = reader.getLmsRearCloud(minTs, maxTs, baseTs);
  std::vector<Vec3> ldmrsCloud = reader.getLdmrsCloud(minTs, maxTs, baseTs);
  std::vector<Vec3> *clouds[totalClouds] = {&lmsFrontCloud, &lmsRearCloud,
                                            &ldmrsCloud};
  sparsify(clouds, totalClouds, FLAGS_cloud_points);

  const cv::Vec3b gray(128, 128, 128);
  std::ofstream ofsFront(FLAGS_out_front);
  printInPly(ofsFront, lmsFrontCloud,
             std::vector<cv::Vec3b>(lmsFrontCloud.size(), gray));
  std::ofstream ofsRear(FLAGS_out_rear);
  printInPly(ofsRear, lmsRearCloud,
             std::vector<cv::Vec3b>(lmsRearCloud.size(), gray));
  std::ofstream ofsLdmrs(FLAGS_out_ldmrs);
  printInPly(ofsLdmrs, ldmrsCloud,
             std::vector<cv::Vec3b>(ldmrsCloud.size(), gray));
}

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  // gflags::SetUsageMessage(usage);
  google::InitGoogleLogging(argv[0]);

  ReaderSettings settings;
  settings.fillVoGaps = FLAGS_fill_vo_gaps;
  settings.projectedTimeWindow = FLAGS_time_win;
  std::optional<fs::path> rtkDir(FLAGS_rtk_dir);
  //  rtkDir.reset();
  RobotcarReader reader(FLAGS_chunk_dir, FLAGS_models_dir, FLAGS_extrinsics_dir,
                        rtkDir, settings);

  Timestamp minTs = reader.minTs();
  Timestamp maxTs = reader.maxTs();
  Timestamp baseTs = minTs;

  if (FLAGS_gen_clouds)
    genClouds(reader, minTs, maxTs, baseTs);

  std::ofstream ofsTrajOrig(FLAGS_out_traj_orig);
  for (const SE3 &bodyToWorld : reader.getGtBodyToWorld()) {
    putInMatrixForm(ofsTrajOrig, bodyToWorld);
  }

  std::ofstream ofsTrajInterp(FLAGS_out_traj_interp);
  Timestamp allTime = maxTs - minTs;
  double step = double(allTime) / FLAGS_interp_gran;
  for (int i = 0; i < FLAGS_interp_gran; ++i) {
    Timestamp ts = minTs + step * i;
    putInMatrixForm(ofsTrajInterp, reader.tsToWorld(ts));
  }

  cv::Mat3b proj = project(reader, FLAGS_project_idx);
  cv::imwrite(FLAGS_out_project, proj);

  cv::Mat3b interp = interpolate(reader, 2000, FLAGS_project_idx);
  cv::imwrite(FLAGS_out_interp, interp);

  return 0;
}
