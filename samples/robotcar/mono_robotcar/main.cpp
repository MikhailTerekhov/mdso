#include "../reader/RobotcarReader.h"
#include "output/CloudWriter.h"
#include "output/CloudWriterGT.h"
#include "output/DebugImageDrawer.h"
#include "output/DepthPyramidDrawer.h"
#include "output/InterpolationDrawer.h"
#include "output/TrackingDebugImageDrawer.h"
#include "output/TrajectoryWriterDso.h"
#include "output/TrajectoryWriterGT.h"
#include "output/TrajectoryWriterPredict.h"
#include "system/DoGPreprocessor.h"
#include "system/DsoSystem.h"
#include "system/IdentityPreprocessor.h"
#include "util/defs.h"
#include "util/flags.h"

DEFINE_string(dataset_dir, "/shared/datasets/oxford-robotcar",
              "Path to the  Oxford Robotcar dataset.");
DEFINE_string(chunk_dir, "2014-05-06-12-54-54",
              "Name of the chunk to output pointcloud from.");
DEFINE_string(models_dir, "data/models/robotcar",
              "Directory with omnidirectional camera models. It is provided in "
              "our repository. IT IS NOT THE \"models\" DIRECTORY FROM THE "
              "ROBOTCAR DATASET SDK!");
DEFINE_string(extrinsics_dir, "thirdparty/robotcar-dataset-sdk/extrinsics",
              "Directory with RobotCar dataset sensor extrinsics, provided in "
              "the dataset SDK.");
DEFINE_string(masks_dir, "data/masks/robotcar",
              "Path to a directory with masks for the dataset.");

DEFINE_int32(start, 1, "Number of the starting frame.");
DEFINE_int32(count, 100, "Number of frames to process.");
DEFINE_string(cam_name, "rear",
              "Name of the camera to use for mono odometry. Possible values "
              "are left, right or rear.");
DEFINE_int32(gt_points, 1'000'000,
             "Number of GT points in the generated cloud.");

DEFINE_bool(gen_gt, true, "Do we need to generate GT pointcloud?");
DEFINE_bool(gen_cloud, true, "Do we need to save resulting pointcloud?");
DEFINE_bool(draw_interpolation, true,
            "Do we need to draw interpolation from initializer?");

DEFINE_bool(use_DoG, false, "Do we need to apply DoG preprocessing first?");
DEFINE_double(sigma1, 0, "Smaller sigma in the DoG filter.");
DEFINE_double(sigma2, 2, "Bigger sigma in  the DoG filter.");
DEFINE_double(DoG_multiplier, 1, "Multiplier after the DoG filter.");

DEFINE_bool(gen_gt_only, false, "Generate ground truth point cloud and exit.");

DEFINE_string(debug_img_dir, "debug",
              "Directory for debug images to be put into.");
DEFINE_string(trajectory_filename, "tracked_frame_to_world.txt",
              "Resulting trajectory filename (stored in 3x4 matrix form).");
DEFINE_string(gt_trajectory_filename, "frame_to_world_GT.txt",
              "Ground truth trajectory filename (stored in 3x4 matrix form).");
DEFINE_string(resulting_cloud_filename, "points.ply", "Output cloud filename.");
DEFINE_string(gt_cloud_filename, "points.ply", "Ground truth cloud filename.");
DEFINE_bool(gen_gt_trajectory, true,
            "Do we need to generate ground truth trajectories?");

DEFINE_string(pred_trajectory_filename, "predicted.txt",
              "Predicted trajectory filename (stored in 3x4 matrix form)");
DEFINE_bool(gen_pred_trajectory, true,
            "Do we need to generate predicted trajectory?");

DEFINE_string(
    track_img_dir, "track",
    "Directory for tracking residuals on all pyr levels to be put into.");
DEFINE_bool(show_track_res, false,
            "Show tracking residuals on all levels of the pyramind?");
DEFINE_bool(show_debug_image, true,
            R"__(Show debug image? Structure of debug image:
+---+---+
| A | B |
+---+---+
| C | D |
+---+---+
Where
A displays color-coded depths projected onto the base frame;
B -- visible vs invizible OptimizedPoint-s projected onto the base frame;
C -- color-coded predicted disparities;
D -- color-coded tracking residuals on the finest pyramid level.)__");
DEFINE_string(depth_pyramid_dir, "pyr",
              "Directory for depth pyramid images to be put into.");
DEFINE_bool(draw_depth_pyramid, false,
            "Draw the depth pyramid, that is used for tracking? Will be "
            "overridden to false if write_files is set to false.");

DEFINE_bool(write_files, true,
            "Do we need to write output files into output_directory?");
DEFINE_string(output_directory, "output/default",
              "DSO's output directory. This flag is disabled, whenever "
              "use_time_for_output is set to true.");
DEFINE_bool(
    use_time_for_output, false,
    "If set to true, output directory is created according to the current "
    "time, instead of using the output_directory flag. The precise format for "
    "the name is output/YYYYMMDD_HHMMSS");

using namespace mdso;

StdVector<SE3> getWorldToFrameVO(const RobotcarReader &reader, int start,
                                 int count, RobotcarReader::CamName camName) {
  const std::vector<Timestamp> &timestamps = reader.camTs(camName);
  CHECK_GE(start, 0);
  CHECK_LT(start, reader.numFrames());
  CHECK_GE(start + count, 0);
  CHECK_LT(start + count, reader.numFrames());
  StdVector<SE3> worldToFrameVO;
  worldToFrameVO.reserve(count);
  for (int frameNum = start; frameNum < start + count; ++frameNum) {
    worldToFrameVO.push_back(
        reader.tsToTs(timestamps[frameNum], timestamps[start]));
  }
  return worldToFrameVO;
}

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  // gflags::SetUsageMessage(usage);
  google::InitGoogleLogging(argv[0]);

  Settings settings = getFlaggedSettings();
  ReaderSettings readerSettings;
  readerSettings.cam = settings.cameraModel;

  fs::path datasetDir(FLAGS_dataset_dir);
  fs::path chunkDir = datasetDir / fs::path(FLAGS_chunk_dir);
  fs::path modelsDir(FLAGS_models_dir);
  fs::path extrinsicsDir(FLAGS_extrinsics_dir);
  fs::path masksDir(FLAGS_masks_dir);
  CHECK(fs::is_directory(chunkDir));
  CHECK(fs::is_directory(modelsDir));
  CHECK(fs::is_directory(extrinsicsDir));
  CHECK(fs::is_directory(masksDir));
  RobotcarReader reader(chunkDir, modelsDir, extrinsicsDir, readerSettings);
  //  reader.provideMasks(masksDir);

  RobotcarReader::CamName camName = RobotcarReader::CAM_REAR;
  if (FLAGS_cam_name == "left")
    camName = RobotcarReader::CAM_LEFT;
  else if (FLAGS_cam_name == "rear")
    camName = RobotcarReader::CAM_REAR;
  else if (FLAGS_cam_name == "right")
    camName = RobotcarReader::CAM_RIGHT;
  else
    LOG(FATAL) << "Unknown camera passed to cam_name flag: \"" << FLAGS_cam_name
               << "\"";

  fs::path outDir(FLAGS_use_time_for_output ? "output/" + curTimeBrief()
                                            : FLAGS_output_directory);
  fs::path debugImgDir = outDir / fs::path(FLAGS_debug_img_dir);
  fs::path trackImgDir = outDir / fs::path(FLAGS_track_img_dir);
  fs::path pyrImgDir = outDir / fs::path(FLAGS_depth_pyramid_dir);
  if (FLAGS_write_files) {
    fs::create_directories(outDir);
    fs::create_directories(debugImgDir);
    fs::create_directories(trackImgDir);
    if (FLAGS_draw_depth_pyramid)
      fs::create_directories(pyrImgDir);
  }

  CameraBundle::CameraEntry camEntry = reader.cam().bundle[camName];
  SE3 bodyToCam = camEntry.bodyToThis;
  SE3 id;
  CameraBundle cameraBundle(&id, &camEntry.cam, 1);
  CameraBundle::CamPyr camPyr =
      cameraBundle.camPyr(settings.pyramid.levelNum());
  std::vector<int> drawingOrder(1, 0);
  TrackingDebugImageDrawer trackingDebugImageDrawer(
      camPyr.data(), settings.frameTracker, settings.pyramid, drawingOrder);
  TrajectoryWriterDso trajectoryWriter(outDir, FLAGS_trajectory_filename);

  StdVector<SE3> frameToWorldGT =
      getWorldToFrameVO(reader, FLAGS_start, FLAGS_count, camName);
  const std::vector<Timestamp> &allTs = reader.camTs(camName);
  std::vector<Timestamp> timestamps(allTs.begin() + FLAGS_start,
                                    allTs.begin() + FLAGS_start + FLAGS_count);
  TrajectoryWriterGT trajectoryWriterGT(frameToWorldGT.data(),
                                        timestamps.data(), timestamps.size(),
                                        outDir, FLAGS_gt_trajectory_filename);

  TrajectoryWriterPredict trajectoryWriterPredict(
      outDir, FLAGS_pred_trajectory_filename);

  std::unique_ptr<CloudWriter> cloudWriter;
  if (FLAGS_gen_cloud)
    cloudWriter = std::unique_ptr<CloudWriter>(
        new CloudWriter(&cameraBundle, outDir, FLAGS_resulting_cloud_filename));

  InterpolationDrawer interpolationDrawer(&cameraBundle.bundle[0].cam);

  DepthPyramidDrawer depthPyramidDrawer;

  Observers observers;
  observers.dso.push_back(&trajectoryWriter);
  if (FLAGS_gen_gt_trajectory)
    observers.dso.push_back(&trajectoryWriterGT);
  if (FLAGS_gen_pred_trajectory)
    observers.dso.push_back(&trajectoryWriterPredict);
  if (FLAGS_gen_cloud)
    observers.dso.push_back(cloudWriter.get());
  if (FLAGS_write_files && FLAGS_draw_depth_pyramid)
    observers.frameTracker.push_back(&depthPyramidDrawer);
  if (FLAGS_write_files || FLAGS_show_track_res)
    observers.frameTracker.push_back(&trackingDebugImageDrawer);
  if (FLAGS_draw_interpolation)
    observers.initializer.push_back(&interpolationDrawer);

  std::cout << "running DSO.." << std::endl;

  LOG(INFO) << "Total DSO observers: " << observers.dso.size()
            << "\nTracker observers: " << observers.frameTracker.size()
            << "\nInit observers: " << observers.initializer.size();

  std::unique_ptr<Preprocessor> preprocessor =
      std::unique_ptr<Preprocessor>(new IdentityPreprocessor());
  DsoSystem dso(&cameraBundle, preprocessor.get(), observers, settings);
  for (int it = FLAGS_start; it < FLAGS_start + FLAGS_count; ++it) {
    std::cout << "add frame #" << it << std::endl;
    auto frame = reader.frame(it);
    dso.addMultiFrame(&frame[camName].frame, &frame[camName].timestamp);

    if (FLAGS_draw_interpolation && interpolationDrawer.didInitialize()) {
      cv::Mat3b interpolation = interpolationDrawer.draw();
      fs::path out = outDir / fs::path("interpolation.jpg");
      cv::imwrite(out.native(), interpolation);
    }

    if (FLAGS_write_files) {
      if (FLAGS_draw_depth_pyramid && depthPyramidDrawer.pyrChanged()) {
        cv::Mat pyrImage = depthPyramidDrawer.getLastPyr();
        fs::path outPyr = fs::path(pyrImgDir) /
                          fs::path("frame#" + std::to_string(it) + ".jpg");
        cv::imwrite(outPyr.native(), pyrImage);
      }
    }
    if (FLAGS_show_track_res)
      cv::imshow("tracking", trackingDebugImageDrawer.drawAllLevels());
    if (FLAGS_show_debug_image || FLAGS_show_track_res)
      cv::waitKey(1);
  }

  return 0;
}
