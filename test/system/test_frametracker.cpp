#include "../../samples/robotcar/reader/RobotcarReader.h"
#include "output/TrackingDebugImageDrawer.h"
#include "system/FrameTracker.h"
#include "system/IdentityPreprocessor.h"
#include "system/TrackingPredictorScrew.h"
#include <gtest/gtest.h>

using namespace mdso;

DEFINE_string(dataset_dir, "/shared/datasets/oxford-robotcar",
              "Path to the  Oxford Robotcar dataset.");

DEFINE_string(
    models_dir, "data/models/robotcar",
    "Path to the directory with RobotCar cameras intrinsic calibration. THIS "
    "IS NOT THE DIRECTORY FROM THE DATASET SDK! We provide our own intrinsic "
    "camera calibration in the repository.");

DEFINE_string(masks_dir, "data/masks/robotcar",
              "Path to a directory with masks for the dataset.");

DEFINE_string(
    extrinsics_dir,
    "/shared/datasets/oxford-robotcar/robotcar-dataset-sdk/extrinsics",
    "Path to a directory with RobotCar dataset extrinsic parameters, as "
    "provided in the dataset SDK.");

DEFINE_double(
    time_win, 5,
    "Lidar scans from the time window [ts - time_win, ts + time_win] will be "
    "used to get the projected cloud. Here ts is the time, corresponding to "
    "project_idx frame. time_win is measured in seconds.");

DEFINE_double(trans_drift, 0.05,
              "Translational drift to be tested against. No units.");
DEFINE_double(rot_drift, 0.02,
              "Rotational drift to be tested against. Measured in deg/m.");

DEFINE_string(output_directory, "output/default",
              "DSO's output directory. This flag is disabled, whenever "
              "use_time_for_output is set to true.");
DEFINE_bool(
    use_time_for_output, false,
    "If set to true, output directory is created according to the current "
    "time, instead of using the output_directory flag. The precise format for "
    "the name is output/YYYYMMDD_HHMMSS");

DEFINE_int32(
    filter_box_size, 3,
    "We leave only one lidar point in a box on the image of this size.");

DEFINE_double(rel_expected_proximity, 0.003,
              "Relative to w+h proximity to lidar point for a contrast point "
              "to be used.");

DEFINE_double(rel_point_size_contr, 0.004,
              "Relative to w+h point size on the images with interpolated "
              "depths in contrast points.");

DEFINE_double(
    rel_point_size_lidar, 0.001,
    "Relative to w+h point size on the images with projected lidar points.");

DEFINE_bool(use_vo_for_prediction, false,
            "If set to true, VO from dataset is used to predict poses prior to "
            "tracking.");

DEFINE_bool(reproj_intercam, true,
            "Do we need to use intercamera reprojections in FrameTracker?");

DEFINE_bool(show_track_res, false,
            "Show tracking residuals on all levels of the pyramind?");

DEFINE_int32(debug_image_w, 1200, "Width of the shown debug image.");

struct TestParams {
  TestParams(const fs::path &chunkDir, Timestamp baseTs, int levelNum,
             int pointsUsed, int framesTracked, bool optimizeAffineLight)
      : chunkDir(chunkDir)
      , baseTs(baseTs)
      , levelNum(levelNum)
      , pointsUsed(pointsUsed)
      , framesTracked(framesTracked)
      , optimizeAffineLight(optimizeAffineLight) {}

  fs::path chunkDir;
  Timestamp baseTs;
  int levelNum;
  int pointsUsed;
  int framesTracked;
  bool optimizeAffineLight;
};

void filterInterpolatablePoints(PixelSelector::PointVector &points,
                                Terrain &terrain, double expectedProximity) {
  auto newEnd = std::remove_if(
      points.begin(), points.end(),
      [&terrain, expectedProximity](const cv::Point &p) {
        double proximity = INF;
        bool isInterpolatable =
            terrain.hasInterpolatedDepth(toVec2(p), proximity);
        return !isInterpolatable || proximity > expectedProximity;
      });
  points.erase(newEnd, points.end());
}

void filterByMask(PixelSelector::PointVector &points, const cv::Mat1b &mask) {
  Eigen::AlignedBox2i bound(Vec2i::Zero(), Vec2i(mask.cols - 1, mask.rows - 1));
  auto newEnd = std::remove_if(points.begin(), points.end(),
                               [&bound, &mask](const cv::Point &p) {
                                 return !bound.contains(toVec2i(p)) || !mask(p);
                               });
  points.erase(newEnd, points.end());
}

void filterOutRepeatedPixels(StdVector<std::pair<Vec2, double>> &depthedPoints,
                             int w, int h, int boxSize) {
  cv::Mat1b mask(h / boxSize, w / boxSize, false);
  Eigen::AlignedBox2i bound(Vec2i::Ones(), Vec2i(mask.cols - 2, mask.rows - 2));
  auto newEnd = std::remove_if(
      depthedPoints.begin(), depthedPoints.end(), [&](const auto &p) {
        Vec2i reduced = (p.first / boxSize).template cast<int>();
        cv::Point reducedCv = toCvPoint(reduced);
        if (bound.contains(reduced) && !mask(reducedCv)) {
          mask(reducedCv) = true;
          return false;
        } else
          return true;
      });
  depthedPoints.erase(newEnd, depthedPoints.end());
}

class FrameTrackerTest : public ::testing::TestWithParam<TestParams> {
protected:
  void SetUp() override {
    LOG(INFO) << "settings up another test";

    int lidarS = FLAGS_rel_point_size_lidar *
                 (RobotcarReader::imageWidth + RobotcarReader::imageHeight) / 2;
    int contrastS = FLAGS_rel_point_size_contr *
                    (RobotcarReader::imageWidth + RobotcarReader::imageHeight) /
                    2;

    outDir = FLAGS_output_directory;
    outDir /= fs::path(
        ::testing::UnitTest::GetInstance()->current_test_info()->name());
    trackDir = outDir / fs::path("track");
    fs::create_directories(outDir);
    fs::create_directories(trackDir);

    fs::path datasetDir(FLAGS_dataset_dir);
    fs::path chunkDir = datasetDir / GetParam().chunkDir;
    baseTs = GetParam().baseTs;
    int levelNum = GetParam().levelNum;
    int pointsUsed = GetParam().pointsUsed;
    framesTracked = GetParam().framesTracked;

    fs::path modelsDir(FLAGS_models_dir);
    fs::path extrinsicsDir(FLAGS_extrinsics_dir);
    fs::path masksDir(FLAGS_masks_dir);
    ASSERT_TRUE(fs::is_directory(chunkDir));
    ASSERT_TRUE(fs::is_directory(modelsDir));
    ASSERT_TRUE(fs::is_directory(extrinsicsDir));
    ASSERT_TRUE(fs::is_directory(masksDir));
    ASSERT_GT(levelNum, 0);
    ASSERT_LT(levelNum, Settings::Pyramid::max_levelNum);
    reader = std::unique_ptr<RobotcarReader>(
        new RobotcarReader(chunkDir, modelsDir, extrinsicsDir));
    reader->provideMasks(masksDir);

    settings.pyramid.setLevelNum(levelNum);
    settings.affineLight.optimizeAffineLight = GetParam().optimizeAffineLight;
    settings.frameTracker.doIntercameraReprojection = FLAGS_reproj_intercam;

    baseInd = std::lower_bound(reader->leftTs().begin(), reader->leftTs().end(),
                               baseTs) -
              reader->leftTs().begin();
    ASSERT_LT(baseInd + framesTracked, reader->leftTs().size());
    auto frame = reader->frame(baseInd);
    cv::Mat1b imgGray[RobotcarReader::numCams];
    cv::Mat1d gradX[RobotcarReader::numCams], gradY[RobotcarReader::numCams],
        gradNorm[RobotcarReader::numCams];
    for (int camInd = 0; camInd < RobotcarReader::numCams; ++camInd) {
      imgGray[camInd] = cvtBgrToGray(frame[camInd].frame);
      grad(imgGray[camInd], gradX[camInd], gradY[camInd], gradNorm[camInd]);
    }

    Timestamp timeWin = FLAGS_time_win * 1e6;
    Timestamp minTs = baseTs - timeWin, maxTs = baseTs + timeWin;
    ASSERT_GT(reader->leftTs()[baseInd], minTs);
    auto projected = reader->project(minTs, maxTs, baseTs);

    FrameTracker::DepthedMultiFrame baseForTracker;
    baseForTracker.reserve(RobotcarReader::numCams);
    cv::Mat3b lidarDepthsIm[RobotcarReader::numCams];
    cv::Mat3b contrastDepthsIm[RobotcarReader::numCams];
    cv::Mat3b terrainsIm[RobotcarReader::numCams];
    for (int camInd = 0; camInd < RobotcarReader::numCams; ++camInd) {
      LOG(INFO) << "for cam #" << camInd << " there are "
                << projected[camInd].size() << " lidar points";
      std::cout << "before filter = " << projected[camInd].size() << std::endl;
      filterOutRepeatedPixels(projected[camInd], frame[camInd].frame.cols,
                              frame[camInd].frame.rows, FLAGS_filter_box_size);
      std::cout << "after filter = " << projected[camInd].size() << std::endl;
      LOG(INFO) << "after filtering out close points: "
                << projected[camInd].size();
      StdVector<Vec2> lidarPoints(projected[camInd].size());
      std::vector<double> lidarDepths(projected[camInd].size());
      for (int i = 0; i < lidarPoints.size(); ++i) {
        lidarPoints[i] = projected[camInd][i].first;
        lidarDepths[i] = projected[camInd][i].second;
      }
      Terrain terrain(&reader->cam().bundle[camInd].cam, lidarPoints,
                      lidarDepths);
      int pointsNeeded = pointsUsed / RobotcarReader::numCams;
      PixelSelector::PointVector contrastPoints = pixelSelectors[camInd].select(
          frame[camInd].frame, gradNorm[camInd], pointsNeeded);
      int pointsUnfiltered = contrastPoints.size();
      int w = frame[camInd].frame.cols, h = frame[camInd].frame.rows;
      double absExpectedProximity = FLAGS_rel_expected_proximity * (w + h);
      LOG(INFO) << "for cam #" << camInd
                << " expected proximity = " << absExpectedProximity;

      filterInterpolatablePoints(contrastPoints, terrain, absExpectedProximity);
      if (reader->masksProvided()) {
        int before = contrastPoints.size();
        filterByMask(contrastPoints, reader->cam().bundle[camInd].cam.mask());
        int diff = before - int(contrastPoints.size());
        LOG(INFO) << "mask filtered out 1 " << diff;
      }
      int pointsFiltered = contrastPoints.size();
      LOG(INFO) << "points filtered: " << pointsFiltered;
      ASSERT_GT(pointsFiltered, 0);
      int newPointsNeeded =
          int(double(pointsUnfiltered) / pointsFiltered * pointsNeeded);
      contrastPoints = pixelSelectors[camInd].select(
          frame[camInd].frame, gradNorm[camInd], newPointsNeeded);
      filterInterpolatablePoints(contrastPoints, terrain, absExpectedProximity);
      if (reader->masksProvided()) {
        int before = contrastPoints.size();
        filterByMask(contrastPoints, reader->cam().bundle[camInd].cam.mask());
        int diff = before - int(contrastPoints.size());
        LOG(INFO) << "mask filtered out 2 " << diff;
      }
      LOG(INFO) << "final points filtered: " << contrastPoints.size();
      StdVector<Vec2> points(contrastPoints.size());
      std::vector<double> depths(contrastPoints.size());
      std::vector<double> weights(contrastPoints.size(), 1);
      for (int i = 0; i < points.size(); ++i) {
        points[i] = toVec2(contrastPoints[i]);
        terrain(points[i], depths[i]);
      }
      baseForTracker.emplace_back(imgGray[camInd], levelNum, points.data(),
                                  depths.data(), weights.data(), points.size());

      if (camInd == 0)
        setDepthColBounds(lidarDepths);
      lidarDepthsIm[camInd] = cvtGrayToBgr(imgGray[camInd]);
      contrastDepthsIm[camInd] = lidarDepthsIm[camInd].clone();
      cv::Mat1b origMask = reader->cam().bundle[camInd].cam.mask();
      cv::Mat3b maskIm = cvtGrayToBgr(origMask);
      VLOG(1) << "orig mask sizes = " << origMask.cols << ' ' << origMask.rows;
      VLOG(1) << "mask sizes = " << maskIm.cols << ' ' << maskIm.rows;
      VLOG(1) << "other sizes = " << contrastDepthsIm[camInd].cols << ' '
              << contrastDepthsIm[camInd].rows;
      //      cv::Mat3b blended;
      //      cv::addWeighted(contrastDepthsIm[camInd], 0.5, maskIm, 0.5, 0,
      //      blended); contrastDepthsIm[camInd] = blended;

      for (int i = 0; i < lidarPoints.size(); ++i)
        putSquare(lidarDepthsIm[camInd], toCvPoint(lidarPoints[i]), lidarS,
                  depthCol(lidarDepths[i], minDepthCol, maxDepthCol),
                  cv::FILLED);
      for (int i = 0; i < points.size(); ++i)
        putSquare(contrastDepthsIm[camInd], toCvPoint(points[i]), contrastS,
                  depthCol(depths[i], minDepthCol, maxDepthCol), cv::FILLED);
      terrainsIm[camInd] = contrastDepthsIm[camInd].clone();
      terrain.draw(terrainsIm[camInd], CV_BLACK, 1);
    }

    cv::Mat3b lidarTotalIm;
    cv::Mat3b contrastTotalIm;
    cv::Mat3b terrainsTotalIm;
    cv::hconcat(lidarDepthsIm, RobotcarReader::numCams, lidarTotalIm);
    cv::hconcat(contrastDepthsIm, RobotcarReader::numCams, contrastTotalIm);
    cv::hconcat(terrainsIm, RobotcarReader::numCams, terrainsTotalIm);
    fs::path lidarDepthsPath = outDir / fs::path("lidar_depths.png");
    fs::path contrastDepthsPath = outDir / fs::path("contrast_depths.png");
    fs::path terrainsPath = outDir / fs::path("terrains.png");
    cv::imwrite(std::string(lidarDepthsPath), lidarTotalIm);
    cv::imwrite(std::string(contrastDepthsPath), contrastTotalIm);
    cv::imwrite(std::string(terrainsPath), terrainsTotalIm);

    camPyr = reader->cam().camPyr(levelNum);
    preprocessor = std::unique_ptr<Preprocessor>(new IdentityPreprocessor);

    std::vector<int> drawingOrder = {0, 2, 1};
    debugImageDrawer.reset(new TrackingDebugImageDrawer(
        camPyr.data(), settings.frameTracker, settings.pyramid, drawingOrder));
    observers.push_back(debugImageDrawer.get());

    cv::Mat3b coloredFrames[RobotcarReader::numCams];
    Timestamp timestamps[RobotcarReader::numCams];
    for (int camInd = 0; camInd < RobotcarReader::numCams; ++camInd) {
      coloredFrames[camInd] = frame[camInd].frame;
      timestamps[camInd] = frame[camInd].timestamp;
    }
    std::unique_ptr<PreKeyFrame> preKeyFrame(
        new PreKeyFrame(nullptr, &camPyr[0], preprocessor.get(), coloredFrames,
                        0, timestamps, settings.pyramid));
    TrackingResult defaultTracking(RobotcarReader::numCams);
    preKeyFrame->setTracked(defaultTracking);
    baseFrame = std::unique_ptr<KeyFrame>(
        new KeyFrame(std::move(preKeyFrame), pixelSelectors.data(),
                     settings.keyFrame, settings.getPointTracerSettings()));
    frameTracker = std::unique_ptr<FrameTracker>(
        new FrameTracker(camPyr.data(), baseForTracker, *baseFrame, observers,
                         settings.getFrameTrackerSettings()));
  }

  fs::path outDir;
  fs::path trackDir;
  Settings settings;
  CameraBundle::CamPyr camPyr;
  std::unique_ptr<RobotcarReader> reader;
  std::unique_ptr<KeyFrame> baseFrame;
  std::unique_ptr<FrameTracker> frameTracker;
  std::unique_ptr<Preprocessor> preprocessor;
  std::unique_ptr<TrackingDebugImageDrawer> debugImageDrawer;
  std::array<PixelSelector, RobotcarReader::numCams> pixelSelectors;
  std::vector<FrameTrackerObserver *> observers;
  Timestamp baseTs;
  int baseInd;
  int framesTracked;
};

template <int numCams> class DummyTrajectoryHolder : public TrajectoryHolder {
public:
  struct Elem {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Elem(const SE3 &thisToWorld, AffLight _lightWorldToThis[],
         Timestamp _timestamps[])
        : thisToWorld(thisToWorld) {
      for (int camInd = 0; camInd < numCams; ++camInd) {
        lightWorldToThis[camInd] = _lightWorldToThis[camInd];
        timestamps[camInd] = _timestamps[camInd];
      }
    }

    SE3 thisToWorld;
    AffLight lightWorldToThis[numCams];
    Timestamp timestamps[numCams];
  };

  int trajectorySize() const override { return elems.size(); }
  int camNumber() const override { return numCams; }
  Timestamp timestamp(int ind) const override {
    CHECK(ind >= 0 && ind < elems.size());
    const Timestamp *ts = elems[ind].timestamps;
    Timestamp sum = std::accumulate(ts, ts + numCams, Timestamp(0));
    return sum / numCams;
  }
  SE3 bodyToWorld(int ind) const override {
    CHECK(ind >= 0 && ind < elems.size());
    return elems[ind].thisToWorld;
  }
  AffLight affLightWorldToBody(int ind, int camInd) const override {
    CHECK(ind >= 0 && ind < elems.size());
    CHECK(camInd >= 0 && camInd < numCams);
    return elems[ind].lightWorldToThis[camInd];
  }

  void pushBack(const SE3 &thisToWorld, AffLight lightWorldToThis[],
                Timestamp timestamps[]) {
    elems.emplace_back(thisToWorld, lightWorldToThis, timestamps);
  }

private:
  StdVector<Elem> elems;
};

TEST_P(FrameTrackerTest, doesTracking) {
  fs::path trackedFile = outDir / fs::path("tracked.txt");
  fs::path predictedFile = outDir / fs::path("predicted.txt");
  fs::path voFile = outDir / fs::path("vo.txt");
  std::ofstream trackedOfs(trackedFile);
  std::ofstream predictedOfs(predictedFile);
  std::ofstream voOfs(voFile);

  putInMatrixForm(trackedOfs, SE3());
  trackedOfs << std::endl;
  putInMatrixForm(predictedOfs, SE3());
  predictedOfs << std::endl;
  putInMatrixForm(voOfs, SE3());
  voOfs << std::endl;

  DummyTrajectoryHolder<RobotcarReader::numCams> trajectoryHolder;
  AffLight defaultAffLight[RobotcarReader::numCams];
  Timestamp baseTimestamps[RobotcarReader::numCams] = {
      reader->leftTs()[baseInd], reader->rearTs()[baseInd],
      reader->rightTs()[baseInd]};
  SE3 idSe3;
  trajectoryHolder.pushBack(idSe3, defaultAffLight, baseTimestamps);
  std::unique_ptr<TrackingPredictor> trackingPredictor(
      new TrackingPredictorScrew(&trajectoryHolder));

  double totalPath = 0;
  double transDrift = INF, rotDrift = INF;
  double sumRightMLeft = 0, sumRearMLeft = 0;
  for (int relInd = 1; relInd < framesTracked - 1; ++relInd) {
    int frameInd = baseInd + relInd;
    auto frame = reader->frame(frameInd);
    cv::Mat3b coloredFrames[RobotcarReader::numCams];
    Timestamp timestamps[RobotcarReader::numCams];
    for (int camInd = 0; camInd < RobotcarReader::numCams; ++camInd) {
      coloredFrames[camInd] = frame[camInd].frame;
      timestamps[camInd] = frame[camInd].timestamp;
    }
    std::unique_ptr<PreKeyFrame> preKeyFrame(
        new PreKeyFrame(baseFrame.get(), &reader->cam(), preprocessor.get(),
                        coloredFrames, relInd, timestamps, settings.pyramid));
    Timestamp avgTs =
        std::accumulate(timestamps, timestamps + RobotcarReader::numCams,
                        Timestamp(0)) /
        RobotcarReader::numCams;
    double rightMLeft = timestamps[2] - timestamps[0];
    double rearMLeft = timestamps[1] - timestamps[0];
    VLOG(1) << "ts right - left  = " << rightMLeft;
    VLOG(1) << "ts rear - left  = " << rearMLeft;
    sumRightMLeft += std::abs(rightMLeft);
    sumRearMLeft += std::abs(rearMLeft);
    TrackingResult coarse = trackingPredictor->predictAt(avgTs, 0);
    SE3 oldPredictedBaseToThis = coarse.baseToTracked;
    SE3 voBaseToTracked = reader->tsToTs(baseTs, reader->leftTs()[frameInd]);
    if (FLAGS_use_vo_for_prediction)
      coarse.baseToTracked = voBaseToTracked;
    TrackingResult trackingResult =
        frameTracker->trackFrame(*preKeyFrame, coarse);

    trajectoryHolder.pushBack(trackingResult.baseToTracked.inverse(),
                              trackingResult.lightBaseToTracked.data(),
                              timestamps);
    SE3 err = trackingResult.baseToTracked * voBaseToTracked.inverse();
    SE3 voCurToPrev = reader->tsToTs(trajectoryHolder.timestamp(relInd),
                                     trajectoryHolder.timestamp(relInd - 1));
    LOG(INFO) << "vo cur to prev trans norm = "
              << voCurToPrev.translation().norm() << ", ts diff = "
              << trajectoryHolder.timestamp(relInd) -
                     trajectoryHolder.timestamp(relInd - 1);
    totalPath += voCurToPrev.translation().norm();
    double transErr = err.translation().norm();
    double rotErr = err.so3().log().norm();
    transDrift = transErr / totalPath;
    rotDrift = rotErr / totalPath;
    LOG(INFO) << "for frame #" << relInd
              << " trans drift = " << transDrift * 100 << "%";
    LOG(INFO) << "for frame #" << relInd << " rot drift = " << rotDrift
              << "deg/m";
    cv::Mat3b debugImage = debugImageDrawer->drawAllLevels();
    fs::path debugImagePath =
        trackDir / (std::to_string(frame[0].timestamp) + ".jpg");
    cv::imwrite(std::string(debugImagePath), debugImage);
    if (FLAGS_show_track_res) {
      cv::Mat3b resized;
      int height =
          double(debugImage.rows) / debugImage.cols * FLAGS_debug_image_w;
      cv::Size size(FLAGS_debug_image_w, height);
      cv::resize(debugImage, resized, size);
      cv::imshow("tracking residuals", resized);
      cv::waitKey(1);
    }

    putInMatrixForm(predictedOfs, oldPredictedBaseToThis.inverse());
    predictedOfs << std::endl;
    putInMatrixForm(trackedOfs, trackingResult.baseToTracked.inverse());
    trackedOfs << std::endl;
    putInMatrixForm(voOfs, voBaseToTracked.inverse());
    voOfs << std::endl;
  }

  LOG(INFO) << "avg |right - left| (s) = "
            << sumRightMLeft / framesTracked / 1e6;
  LOG(INFO) << "avg |rear - left| (s) = " << sumRearMLeft / framesTracked / 1e6;

  EXPECT_LE(transDrift, FLAGS_trans_drift);
  EXPECT_LE(rotDrift, FLAGS_rot_drift);
}

std::vector<TestParams> getParams() {
  Timestamp baseTimestamps[] = {1447410513767257ll};
  fs::path chunkDir("2015-11-13-10-28-08");
  //  Timestamp baseTimestamps[] = {1399381498321897ll, 1399381506388605ll,
  //                                1399381546822399ll, 1399381565109330ll};
  //  fs::path chunkDir("2014-05-06-12-54-54");
  const int levelNum = 4;
  const int pointsNum = 3000;
  const int framesToTrack = 30;
  bool useAffLight = true;
  std::vector<TestParams> params;
  for (Timestamp ts : baseTimestamps)
    params.push_back(TestParams(chunkDir, ts, levelNum, pointsNum,
                                framesToTrack, useAffLight));
  return params;
}

INSTANTIATE_TEST_CASE_P(Instantiation, FrameTrackerTest,
                        ::testing::ValuesIn(getParams()));

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  fs::path outDir =
      fs::path(FLAGS_use_time_for_output ? "output/" + curTimeBrief()
                                         : FLAGS_output_directory);
  LOG(INFO) << "output will be in " << std::string(outDir);
  FLAGS_output_directory = std::string(outDir);

  return RUN_ALL_TESTS();
}
