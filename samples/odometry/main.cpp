#include "data/MultiCamReader.h"
#include "data/MultiFovReader.h"
#include "data/RobotcarReader.h"
#include "data/SingleCamProxyReader.h"
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
#include "system/DsoInitializerGroundTruth.h"
#include "system/DsoSystem.h"
#include "system/IdentityPreprocessor.h"
#include "util/defs.h"
#include "util/flags.h"
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>

using namespace mdso;

DEFINE_string(robotcar_chunk_dir, "2015-02-10-11-58-05",
              "Chunk of the Oxford Robotcar dataset that is to be used.");
DEFINE_string(robotcar_rtk_dir, "/shared/datasets/oxford-robotcar/rtk",
              "Path to the RTK ground truth trajectories directory.");
DEFINE_string(robotcar_models_dir, "data/models/robotcar",
              "Directory with omnidirectional camera models. It is provided in "
              "our repository. IT IS NOT THE \"models\" DIRECTORY FROM THE "
              "ROBOTCAR DATASET SDK!");
DEFINE_string(robotcar_extrinsics_dir,
              "thirdparty/robotcar-dataset-sdk/extrinsics",
              "Directory with RobotCar dataset sensor extrinsics, provided in "
              "the dataset SDK.");
DEFINE_string(robotcar_masks_dir, "data/masks/robotcar",
              "Path to a directory with masks for the dataset.");

DEFINE_bool(mcam_lift_front, false,
            "If set to true, in MultiCamReader height of the front camera is "
            "aligned with that of the rear.");
DEFINE_bool(mcam_gt_depths, false,
            "If set to false, in MultiCamReader depths are interpolated from "
            "ORB keypoints.");

DEFINE_int32(
    start, 1,
    "Number of the starting frame. This flag is disabled if start_ts is set.");
DEFINE_int64(start_ts, -1,
             "If set to a positive value, determines the closest timestamp to "
             "the starting frame.");

DEFINE_int32(count, 100, "Number of frames to process.");

DEFINE_bool(gen_gt, false, "Do we need to generate GT pointcloud?");
DEFINE_int64(gt_points, 1'000'000,
             "Number of GT points in the generated cloud.");
DEFINE_bool(gen_gt_only, false, "Generate ground truth point cloud and exit.");
DEFINE_bool(cloud_Nx6, false,
            "Generate ground truth point cloud in Nx6 binary format.");
DEFINE_bool(gen_cloud, true, "Do we need to save resulting pointcloud?");

DEFINE_bool(draw_interpolation, false,
            "Do we need to draw interpolation from delaunayInitializer?");

DEFINE_bool(use_DoG, false, "Do we need to apply DoG preprocessing first?");
DEFINE_double(sigma1, 0, "Smaller sigma in the DoG filter.");
DEFINE_double(sigma2, 2, "Bigger sigma in  the DoG filter.");
DEFINE_double(DoG_multiplier, 1, "Multiplier after the DoG filter.");

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
              "Predicted trajectory filename (stored in 3x4 matrix form).");
DEFINE_bool(gen_pred_trajectory, true,
            "Do we need to generate predicted trajectory?");

DEFINE_string(
    track_img_dir, "track",
    "Directory for tracking residuals on all pyr levels to be put into.");
DEFINE_bool(show_track_res, false,
            "Show tracking residuals on all levels of the pyramind?");
DEFINE_bool(show_debug_image, true,
            R"__(Show debug image? Structure of the debug image:
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
            "in MultiFoV dataset is added.");

DEFINE_int32(
    only_camera, -1,
    "If set to a non-negative value, this flag restricts the camera bundle to "
    "a single camera. In that case, flag value is the index of this camera.");

DEFINE_bool(gt_init, false, "Use ground-truth DSO initializer?");

DEFINE_double(scale, 1, "Scale of the dataset used to adjust the settings.");

std::unique_ptr<DatasetReader> createReader(const fs::path &datasetDir) {
  if (MultiFovReader::isMultiFov(datasetDir)) {
    return std::unique_ptr<DatasetReader>(new MultiFovReader(datasetDir));
  } else if (MultiCamReader::isMultiCam(datasetDir)) {
    MultiCamReader::Settings settings;
    settings.useInterpolatedDepths = !FLAGS_mcam_gt_depths;
    settings.numKeyPoints = 200;
    if (FLAGS_mcam_lift_front)
      settings.camNames = {"front_lifted", "right", "rear", "left"};
    return std::unique_ptr<DatasetReader>(
        new MultiCamReader(datasetDir, settings));
  } else if (fs::path chunkDir = datasetDir / FLAGS_robotcar_chunk_dir;
             RobotcarReader::isRobotcar(chunkDir)) {
    std::optional<fs::path> rtkDir;
    if (fs::exists(FLAGS_robotcar_rtk_dir))
      rtkDir.emplace(FLAGS_robotcar_rtk_dir);
    ReaderSettings readerSettings;
    auto robotcarReader =
        new RobotcarReader(chunkDir, FLAGS_robotcar_models_dir,
                           FLAGS_robotcar_extrinsics_dir, rtkDir);
    if (fs::exists(FLAGS_robotcar_masks_dir))
      robotcarReader->provideMasks(FLAGS_robotcar_masks_dir);
    return std::unique_ptr<DatasetReader>(robotcarReader);
  } else {
    LOG(FATAL) << "Unknown dataset on " << datasetDir << " was provided.";
    return nullptr;
  }
}

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

void readPointsInFrameGT(const DatasetReader *reader,
                         std::vector<std::vector<Vec3>> &pointsInFrameGT,
                         std::vector<std::vector<cv::Vec3b>> &colors,
                         int64_t maxPoints) {
  CameraBundle cam = reader->cam();
  std::cout << "filling GT points..." << std::endl;
  int w = cam.bundle[0].cam.getWidth(), h = cam.bundle[0].cam.getHeight();
  int step = std::ceil(std::sqrt(double(FLAGS_count) * w * h *
                                 cam.bundle.size() / FLAGS_gt_points));
  std::cout << "step = " << step << std::endl;

  const double maxd = 1e5;
  for (int it = FLAGS_start; it < FLAGS_start + FLAGS_count; ++it) {
    pointsInFrameGT[it].reserve((h / step) * (w / step));
    auto depths = reader->depths(it);
    auto frame = reader->frame(it);
    for (int ci = 0; ci < cam.bundle.size(); ++ci) {
      for (int y = 0; y < h; y += step)
        for (int x = 0; x < w; x += step) {
          Vec3 p = cam.bundle[ci].cam.unmap(Vec2(x, y));
          p.normalize();
          auto maybeD = depths->depth(ci, Vec2(x, y));
          if (maybeD) {
            double d = maybeD.value();
            if (d > maxd)
              continue;
            p *= d;
            pointsInFrameGT[it].push_back(cam.bundle[ci].thisToBody * p);
            colors[it].push_back(frame[ci].frame(y, x));
          }
        }
    }
    std::cout << "processed frame #" << it << std::endl;
  }
}

void mkdir(const fs::path &dirname) { fs::create_directories(dirname); }

std::pair<std::vector<cv::Mat3b>, std::vector<Timestamp>>
cvtFrame(const std::vector<DatasetReader::FrameEntry> &frame) {
  std::pair<std::vector<cv::Mat3b>, std::vector<Timestamp>> result;
  result.first.reserve(frame.size());
  result.second.reserve(frame.size());
  for (int i = 0; i < frame.size(); ++i) {
    result.first.push_back(frame[i].frame);
    result.second.push_back(frame[i].timestamp);
  }
  return result;
}

cv::Mat3b drawIntialization(const std::vector<const KeyFrame *> &keyFrames,
                            const Settings::Depth &depthSettings,
                            bool includeReproj = true,
                            bool reprojOnOther = true) {

  CameraBundle *cam = keyFrames[0]->preKeyFrame->cam;

  Settings::Depth ds = depthSettings;
  ds.max /= 1e2;
  ds.min /= 1e2;
  int numCameras = keyFrames[0]->preKeyFrame->cam->bundle.size();
  std::vector<std::vector<cv::Mat3b>> frames(keyFrames.size());
  for (int kfInd = 0; kfInd < keyFrames.size(); ++kfInd) {
    frames[kfInd].resize(numCameras);
    auto keyFrame = keyFrames[kfInd];
    for (int camInd = 0; camInd < numCameras; ++camInd)
      frames[kfInd][camInd] =
          keyFrame->preKeyFrame->frames[camInd].frameColored.clone();

    if (includeReproj) {
      auto kfPtr = reprojOnOther ? keyFrames.data() : &keyFrames[kfInd];
      int kfSize = reprojOnOther ? keyFrames.size() : 1;
      Reprojector<ImmaturePoint> reprojector(kfPtr, kfSize,
                                             keyFrame->thisToWorld(), ds);
      auto reprojections = reprojector.reproject();
      if (!isDepthColSet) {
        std::vector<double> depths;
        for (const Reprojection &r : reprojections)
          depths.push_back(r.reprojectedDepth);
        setDepthColBounds(depths);
      }
      for (const Reprojection &r : reprojections) {
        cv::circle(frames[kfInd][r.targetCamInd], toCvPoint(r.reprojected), 3,
                   depthCol(r.reprojectedDepth, minDepthCol, maxDepthCol),
                   cv::FILLED);
      }
    } else {
      if (!isDepthColSet) {
        std::vector<double> depths;
        for (int camInd = 0; camInd < numCameras; ++camInd)
          for (const auto &ip : keyFrame->frames[camInd].immaturePoints)
            depths.push_back(ip.depth);
        setDepthColBounds(depths);
      }
      for (int camInd = 0; camInd < numCameras; ++camInd)
        for (const auto &ip : keyFrame->frames[camInd].immaturePoints)
          cv::circle(frames[kfInd][camInd], toCvPoint(ip.p), 3,
                     depthCol(ip.depth, minDepthCol, maxDepthCol), cv::FILLED);
    }
  }
  std::vector<cv::Mat3b> kfDrawn(keyFrames.size());
  for (int kfInd = 0; kfInd < keyFrames.size(); ++kfInd)
    cv::hconcat(frames[kfInd].data(), frames[kfInd].size(), kfDrawn[kfInd]);
  cv::Mat result;
  cv::vconcat(kfDrawn.data(), kfDrawn.size(), result);
  return result;
}

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

  std::unique_ptr<DatasetReader> reader = createReader(argv[1]);
  if (FLAGS_only_camera >= 0) {
    CHECK_LT(FLAGS_only_camera, reader->cam().bundle.size());
    std::unique_ptr<DatasetReader> restrictedReader(
        new SingleCamProxyReader(std::move(reader), FLAGS_only_camera));
    reader = std::move(restrictedReader);
  }

  Settings settings = getFlaggedSettings();
  CHECK_GT(FLAGS_scale, 0);
  settings = settings.getScaleAdjustedSettings(FLAGS_scale);

  IdentityPreprocessor idPreprocessor;
  DoGPreprocessor dogPreprocessor(FLAGS_sigma1, FLAGS_sigma2,
                                  FLAGS_DoG_multiplier);
  Preprocessor *preprocessor =
      FLAGS_use_DoG ? static_cast<Preprocessor *>(&dogPreprocessor)
                    : static_cast<Preprocessor *>(&idPreprocessor);

  if (FLAGS_use_DoG) {
    cv::Mat3b img = reader->frame(FLAGS_start)[0].frame;
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
    std::vector<std::vector<Vec3>> pointsInFrameGT(reader->numFrames());
    std::vector<std::vector<cv::Vec3b>> colors(reader->numFrames());
    readPointsInFrameGT(reader.get(), pointsInFrameGT, colors, FLAGS_gt_points);

    std::cout << "points read\n";
    std::vector<Vec3> allPoints;
    std::vector<cv::Vec3b> allColors;
    allPoints.reserve(FLAGS_gt_points);
    allColors.reserve(FLAGS_gt_points);
    for (int i = 0; i < pointsInFrameGT.size(); ++i) {
      for (int j = 0; j < pointsInFrameGT[i].size(); ++j) {
        const Vec3 &p = pointsInFrameGT[i][j];
        allPoints.push_back(reader->frameToWorld(i) * p);
        allColors.push_back(colors[i][j]);
      }
      if (i % 100 == 0)
        std::cout << "processed frame #" << i << std::endl;
    }
    if (FLAGS_cloud_Nx6) {
      printInBinNx6(outDir / "pointsGT.bin", allPoints, allColors);
    } else {
      std::ofstream pointsGTOfs(outDir / "pointsGT.ply");
      printInPly(pointsGTOfs, allPoints, allColors);
    }
    return 0;
  }

  TrajectoryWriterDso trajectoryWriter(outDir / FLAGS_trajectory_filename);
  std::optional<TrajectoryWriterGT> trajectoryWriterGT;
  if (FLAGS_gen_gt_trajectory)
    trajectoryWriterGT.emplace(reader.get(), outDir,
                               FLAGS_gt_trajectory_filename);
  TrajectoryWriterPredict trajectoryWriterPredict(
      outDir, FLAGS_pred_trajectory_filename);

  CameraBundle cam = reader->cam();
  if (FLAGS_move_body_from_camera) {
    CHECK_EQ(cam.bundle.size(), 1);
    std::mt19937 mt;
    SE3 camToBody = SE3::sampleUniform(mt);
    cam.setCamToBody(0, camToBody);
    trajectoryWriter.outputModeFrameToWorld(camToBody);
    trajectoryWriterPredict.outputModeFrameToWorld(camToBody);
  }
  CameraBundle::CamPyr camPyr = cam.camPyr(settings.pyramid.levelNum());

  std::vector<int> drawingOrder(cam.bundle.size(), 0);
  for (int i = 0; i < cam.bundle.size(); ++i)
    drawingOrder[i] = i;
  DebugImageDrawer debugImageDrawer(drawingOrder);
  TrackingDebugImageDrawer trackingDebugImageDrawer(
      camPyr.data(), settings.frameTracker, settings.pyramid, drawingOrder);

  std::unique_ptr<CloudWriter> cloudWriter;
  if (FLAGS_gen_cloud)
    cloudWriter = std::unique_ptr<CloudWriter>(
        new CloudWriter(&cam, outDir, FLAGS_resulting_cloud_filename));

  std::unique_ptr<CloudWriterGT> cloudWriterGTPtr;
  //  if (FLAGS_gen_gt) {
  //    std::vector<std::vector<Vec3>> pointsInFrameGT(reader->numFrames());
  //    std::vector<std::vector<cv::Vec3b>> colors(reader->numFrames());
  //    readPointsInFrameGT(reader.get(), pointsInFrameGT, colors,
  //    FLAGS_gt_points); cloudWriterGTPtr.reset(new CloudWriterGT(
  //        frameToWorldGT.data(), timestamps.data(), pointsInFrameGT.data(),
  //        colors.data(), reader->numFrames(), outDir,
  //        FLAGS_gt_cloud_filename));
  //  }

  InterpolationDrawer interpolationDrawer(&cam.bundle[0].cam);

  DepthPyramidDrawer depthPyramidDrawer;

  Observers observers;
  if (FLAGS_write_files || FLAGS_show_debug_image)
    observers.dso.push_back(&debugImageDrawer);
  observers.dso.push_back(&trajectoryWriter);
  if (FLAGS_gen_gt_trajectory && trajectoryWriterGT)
    observers.dso.push_back(&*trajectoryWriterGT);
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

  std::unique_ptr<DsoInitializer> dsoInitializer;
  if (cam.bundle.size() > 1 || FLAGS_gt_init) {
    LOG(INFO) << "using ground truth intializer";
    dsoInitializer.reset(new DsoInitializerGroundTruth(
        reader.get(), settings.getInitializerGroundTruthSettings()));
  }

  int startInd = FLAGS_start;
  if (FLAGS_start_ts > 0) {
    startInd = reader->firstTimestampToInd(FLAGS_start_ts);
  }

  std::cout << "running DSO.." << std::endl;

  LOG(INFO) << "Total DSO observers: " << observers.dso.size()
            << "\nTracker observers: " << observers.frameTracker.size()
            << "\nInit observers: " << observers.initializer.size();

  {
    DsoSystem dso(&cam, preprocessor, observers, settings,
                  std::move(dsoInitializer));
    bool wasInitialized = false;
    for (int it = startInd; it < startInd + FLAGS_count; ++it) {
      std::cout << "add frame #" << it << std::endl;
      auto [frames, timestamps] = cvtFrame(reader->frame(it));
      dso.addMultiFrame(frames.data(), timestamps.data());
      if (FLAGS_draw_interpolation && interpolationDrawer.didInitialize()) {
        cv::Mat3b interpolation = interpolationDrawer.draw();
        fs::path out = outDir / fs::path("interpolation.jpg");
        cv::imwrite(out.native(), interpolation);
      }
      if (!wasInitialized && dso.getIsInitialized()) {
        wasInitialized = true;
        cv::Mat3b initialized =
            drawIntialization(dso.getKeyFrames(), settings.depth, true, true);
        cv::imwrite((outDir / "initialized.jpg").string(), initialized);
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
