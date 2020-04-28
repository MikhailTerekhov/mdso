#include "data/MultiFovReader.h"
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
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>

using namespace mdso;

DEFINE_int32(start, 1, "Number of the starting frame.");
DEFINE_int32(count, 100, "Number of frames to process.");
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
DEFINE_string(timestamps_filename, "timestamps.txt",
              "Timestamps in the resulting trajectory filename.");
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
DEFINE_string(dir_prefix, "", "The prefix added to distinguish the output");
DEFINE_bool(
    use_time_for_output, true,
    "If set to true, output directory is created according to the current "
    "time, instead of using the output_directory flag. The precise format for "
    "the name is output/YYYYMMDD_HHMMSS");

DEFINE_bool(move_body_from_camera, false,
            "If set to true, nontrivial motion between camera body and image "
            "is added.");

class TrackingErrorCollector : public FrameTrackerObserver {
public:
  TrackingErrorCollector(int numLevels)
      : avgRes(numLevels) {}

  void levelTracked(int pyrLevel, const TrackingResult &result,
                    const std::vector<StdVector<std::pair<Vec2, double>>>
                        &pointResiduals) override {
    double sumRes = 0;
    int numRes = 0;
    for (const auto &v : pointResiduals) {
      numRes += v.size();
      for (const auto &[p, r] : v)
        sumRes += r * r;
    }
    avgRes[pyrLevel].push_back(sumRes / numRes);
  }

  void saveErrors(const fs::path &errorsFname) {
    std::ofstream ofs(errorsFname);
    for (const auto &v : avgRes) {
      for (const auto &r : v)
        ofs << r << ' ';
      ofs << '\n';
    }
  }

private:
  std::vector<std::vector<double>> avgRes;
};

void readPointsInFrameGT(const MultiFovReader &reader,
                         std::vector<std::vector<Vec3>> &pointsInFrameGT,
                         std::vector<std::vector<cv::Vec3b>> &colors,
                         int maxPoints) {
  CameraModel cam = reader.cam().bundle[0].cam;
  std::cout << "filling GT points..." << std::endl;
  int w = cam.getWidth(), h = cam.getHeight();
  int step =
      std::ceil(std::sqrt(double(FLAGS_count) * w * h / FLAGS_gt_points));
  const double maxd = 1e10;
  for (int it = FLAGS_start; it < FLAGS_start + FLAGS_count; ++it) {
    pointsInFrameGT[it].reserve((h / step) * (w / step));
    auto depths = reader.depths(it);
    cv::Mat3b frame = reader.frame(it)[0].frame;
    for (int y = 0; y < h; y += step)
      for (int x = 0; x < w; x += step) {
        Vec3 p = cam.unmap(Vec2(x, y));
        p.normalize();
        auto maybeD = depths->depth(0, Vec2(x, y));
        if (maybeD) {
          double d = maybeD.value();
          if (d > maxd)
            continue;
          p *= d;
          pointsInFrameGT[it].push_back(p);
          colors[it].push_back(frame(y, x));
        }
      }
  }
}

void mkdir(const fs::path &dirname) { fs::create_directories(dirname); }

int main(int argc, char **argv) {
  std::string usage = "Usage: " + std::string(argv[0]) + R"abacaba( data_dir
Where data_dir names a directory with MultiFoV fishseye dataset.
It should contain "info" and "data" subdirectories.)abacaba";

  std::vector<std::string> argsVec;
  argsVec.reserve(argc);
  for (int i = 0; i < argc; ++i)
    argsVec.emplace_back(argv[i]);

  gflags::ParseCommandLineFlags(&argc, &argv, true);
  gflags::SetUsageMessage(usage);
  google::InitGoogleLogging(argv[0]);

  if (argc != 2) {
    std::cerr << "Wrong number of arguments!\n" << usage << std::endl;
    return 1;
  }

  fs::path outDir(FLAGS_use_time_for_output
                      ? fs::path("output") / FLAGS_dir_prefix / curTimeBrief()
                      : fs::path(FLAGS_output_directory));
  fs::path debugImgDir = outDir / fs::path(FLAGS_debug_img_dir);
  fs::path trackImgDir = outDir / fs::path(FLAGS_track_img_dir);
  fs::path pyrImgDir = outDir / fs::path(FLAGS_depth_pyramid_dir);

  if (FLAGS_write_files) {
    mkdir(outDir);
    mkdir(debugImgDir);
    mkdir(trackImgDir);
    if (FLAGS_draw_depth_pyramid)
      mkdir(pyrImgDir);

    std::ofstream argsOfs(outDir / "args.txt");
    for (const std::string &a : argsVec)
      argsOfs << a << "\n";
  }

  MultiFovReader reader(argv[1]);

  Settings settings = getFlaggedSettings();

  IdentityPreprocessor idPreprocessor;
  DoGPreprocessor dogPreprocessor(FLAGS_sigma1, FLAGS_sigma2,
                                  FLAGS_DoG_multiplier);
  Preprocessor *preprocessor =
      FLAGS_use_DoG ? static_cast<Preprocessor *>(&dogPreprocessor)
                    : static_cast<Preprocessor *>(&idPreprocessor);

  if (FLAGS_use_DoG) {
    cv::Mat3b img = reader.frame(FLAGS_start)[0].frame;
    cv::Mat1b imgGray = cvtBgrToGray(img);
    cv::Mat1d gradX, gradY, gradNorm;
    grad(imgGray, gradX, gradY, gradNorm);
    cv::Mat1b imgDog;
    preprocessor->process(&imgGray, &imgDog, 1);
    cv::Mat1d gradDogX, gradDogY, gradDogNorm;
    grad(imgDog, gradDogX, gradDogY, gradDogNorm);
    uint8_t meanOrig = cv::mean(imgGray)[0];
    uint8_t meanDog = cv::mean(imgDog)[0];
    double meanGradOrig = cv::mean(gradNorm)[0];
    double meanGradDog = cv::mean(gradDogNorm)[0];
    settings = settings.getGradientAdjustedSettings(
        double(meanDog) / meanOrig, double(meanGradDog) / meanGradOrig);
    settings.pyramid.setLevelNum(4);

    LOG(INFO) << "mean grad norm original = " << meanGradOrig
              << ", DoG = " << meanGradDog
              << "DoG / orig = " << double(meanGradDog) / meanGradOrig;
    LOG(INFO) << "new outlier intensity diff = "
              << settings.intensity.outlierDiff;
    LOG(INFO) << "mean intensity after DoG = " << int(meanDog);
    std::cout << "mean int = " << int(meanDog);
  }

  if (FLAGS_gen_gt_only) {
    std::vector<std::vector<Vec3>> pointsInFrameGT(reader.numFrames());
    std::vector<std::vector<cv::Vec3b>> colors(reader.numFrames());
    readPointsInFrameGT(reader, pointsInFrameGT, colors, FLAGS_gt_points);
    std::vector<Vec3> allPoints;
    std::vector<cv::Vec3b> allColors;
    allPoints.reserve(FLAGS_gt_points);
    allColors.reserve(FLAGS_gt_points);
    for (int i = 0; i < pointsInFrameGT.size(); ++i)
      for (int j = 0; j < pointsInFrameGT[i].size(); ++j) {
        const Vec3 &p = pointsInFrameGT[i][j];
        allPoints.push_back(reader.frameToWorld(i) * p);
        allColors.push_back(colors[i][j]);
      }
    std::ofstream pointsGTOfs("pointsGT.ply");
    printInPly(pointsGTOfs, allPoints, allColors);
    return 0;
  }

  TrajectoryWriterDso trajectoryWriter(outDir / FLAGS_trajectory_filename);
  StdVector<SE3> frameToWorldGT(reader.numFrames());
  std::vector<Timestamp> timestamps(reader.numFrames());
  for (int i = 0; i < timestamps.size(); ++i) {
    timestamps[i] = i;
    frameToWorldGT[i] = reader.frameToWorld(i);
  }
  TrajectoryWriterGT trajectoryWriterGT(frameToWorldGT.data(),
                                        timestamps.data(), timestamps.size(),
                                        outDir, FLAGS_gt_trajectory_filename);
  TrajectoryWriterPredict trajectoryWriterPredict(
      outDir, FLAGS_pred_trajectory_filename);

  CameraBundle cam = reader.cam();
  if (FLAGS_move_body_from_camera) {
    CHECK_EQ(cam.bundle.size(), 1);
    std::mt19937 mt;
    SE3 camToBody = SE3::sampleUniform(mt);
    cam.setCamToBody(0, camToBody);
    trajectoryWriter.outputModeFrameToWorld(camToBody);
    trajectoryWriterPredict.outputModeFrameToWorld(camToBody);
  }
  CameraBundle::CamPyr camPyr = cam.camPyr(settings.pyramid.levelNum());

  std::vector<int> drawingOrder(1, 0);
  DebugImageDrawer debugImageDrawer(drawingOrder);
  TrackingDebugImageDrawer trackingDebugImageDrawer(
      camPyr.data(), settings.frameTracker, settings.pyramid, drawingOrder);

  std::unique_ptr<CloudWriter> cloudWriter;
  if (FLAGS_gen_cloud)
    cloudWriter = std::unique_ptr<CloudWriter>(
        new CloudWriter(&cam, outDir, FLAGS_resulting_cloud_filename));

  std::unique_ptr<CloudWriterGT> cloudWriterGTPtr;
  if (FLAGS_gen_gt) {
    std::vector<std::vector<Vec3>> pointsInFrameGT(reader.numFrames());
    std::vector<std::vector<cv::Vec3b>> colors(reader.numFrames());
    readPointsInFrameGT(reader, pointsInFrameGT, colors, FLAGS_gt_points);
    cloudWriterGTPtr.reset(new CloudWriterGT(
        frameToWorldGT.data(), timestamps.data(), pointsInFrameGT.data(),
        colors.data(), reader.numFrames(), outDir, FLAGS_gt_cloud_filename));
  }

  InterpolationDrawer interpolationDrawer(&cam.bundle[0].cam);

  DepthPyramidDrawer depthPyramidDrawer;

  Observers observers;
  if (FLAGS_write_files || FLAGS_show_debug_image)
    observers.dso.push_back(&debugImageDrawer);
  observers.dso.push_back(&trajectoryWriter);
  if (FLAGS_gen_gt_trajectory)
    observers.dso.push_back(&trajectoryWriterGT);
  if (FLAGS_gen_pred_trajectory)
    observers.dso.push_back(&trajectoryWriterPredict);
  if (FLAGS_gen_cloud)
    observers.dso.push_back(cloudWriter.get());
  if (FLAGS_write_files && FLAGS_draw_depth_pyramid)
    observers.frameTracker.push_back(&depthPyramidDrawer);
  if (cloudWriterGTPtr)
    observers.dso.push_back(cloudWriterGTPtr.get());
  if (FLAGS_write_files || FLAGS_show_track_res)
    observers.frameTracker.push_back(&trackingDebugImageDrawer);
  if (FLAGS_draw_interpolation)
    observers.initializer.push_back(&interpolationDrawer);

  TrackingErrorCollector errorCollector(settings.pyramid.levelNum());
  observers.frameTracker.push_back(&errorCollector);

  std::cout << "running DSO.." << std::endl;

  LOG(INFO) << "Total DSO observers: " << observers.dso.size()
            << "\nTracker observers: " << observers.frameTracker.size()
            << "\nInit observers: " << observers.initializer.size();

  {
    DsoSystem dso(&cam, preprocessor, observers, settings);
    for (int it = FLAGS_start; it < FLAGS_start + FLAGS_count; ++it) {
      std::cout << "add frame #" << it << std::endl;
      cv::Mat3b frame = reader.frame(it)[0].frame;
      Timestamp timestamp = it;
      dso.addMultiFrame(&frame, &timestamp);

      if (FLAGS_draw_interpolation && interpolationDrawer.didInitialize()) {
        cv::Mat3b interpolation = interpolationDrawer.draw();
        fs::path out = outDir / fs::path("interpolation.jpg");
        cv::imwrite(out.native(), interpolation);
      }

      if (FLAGS_write_files) {
        if (debugImageDrawer.isDrawable()) {
          cv::Mat3b debugImage = debugImageDrawer.draw();
          fs::path outDeb =
              debugImgDir / fs::path("frame#" + std::to_string(it) + ".jpg");
          cv::imwrite(outDeb.native(), debugImage);
        }
        if (trackingDebugImageDrawer.isDrawable()) {
          cv::Mat3b trackImage = trackingDebugImageDrawer.drawAllLevels();
          fs::path outTrack =
              trackImgDir / fs::path("frame#" + std::to_string(it) + ".jpg");
          cv::imwrite(outTrack.native(), trackImage);
        }
        if (FLAGS_draw_depth_pyramid && depthPyramidDrawer.pyrChanged()) {
          cv::Mat pyrImage = depthPyramidDrawer.getLastPyr();
          fs::path outPyr = fs::path(pyrImgDir) /
                            fs::path("frame#" + std::to_string(it) + ".jpg");
          cv::imwrite(outPyr.native(), pyrImage);
        }
      }
      if (FLAGS_show_debug_image || FLAGS_show_track_res)
        cv::waitKey(1);
    }
  }

  trajectoryWriter.saveTimestamps(outDir / fs::path(FLAGS_timestamps_filename));
  errorCollector.saveErrors(outDir / fs::path("residuals_track.txt"));

  return 0;
}