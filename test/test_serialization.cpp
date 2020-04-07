#include "../samples/mfov/reader/MultiFovReader.h"
#include "output/TrajectoryWriter.h"
#include "system/DsoSystem.h"
#include "system/serialization.h"
#include "util/flags.h"
#include <gtest/gtest.h>

using namespace fishdso;

DEFINE_string(mfov_dir, "/shared/datasets/mfov",
              "Path to the MultiFoV dataset.");
DEFINE_string(traj_original, "traj_clean.txt",
              "Name of the file with the original trajectory.");
DEFINE_string(traj_restored, "traj_restored.txt",
              "Name of the file with the trajectory after the restoration.");
DEFINE_int32(start, 375, "Number of the starting frame.");
DEFINE_int32(count_before_interruption, 100,
             "Number of the frame after which the interruption happens.");
DEFINE_int32(count, 200, "Total number of frames to process.");

Observers createObservers(TrajectoryWriter &trajectoryWriter) {
  Observers observers;
  observers.dso.push_back(&trajectoryWriter);
  return observers;
}

class SerializationTest : public ::testing::Test {
public:
  static constexpr double maxTransErr = 1e-2;
  static constexpr double maxRotErr = 0.1;

  SerializationTest()
      : outDir(fs::path("output") / curTimeBrief())
      , datasetReader(new MultiFovReader(FLAGS_mfov_dir))
      , cam(*datasetReader->cam)
      , settings(getFlaggedSettings())
      , trajectoryWriterClean(outDir, "oldstyle_traj.txt", FLAGS_traj_original)
      , originalObservers(createObservers(trajectoryWriterClean))
      , dsoOriginal(new DsoSystem(&cam, originalObservers, settings)) {
    fs::create_directories(outDir);
  }

protected:
  fs::path outDir;
  std::unique_ptr<MultiFovReader> datasetReader;
  CameraModel cam;
  Settings settings;
  TrajectoryWriter trajectoryWriterClean;
  Observers originalObservers;
  std::unique_ptr<DsoSystem> dsoOriginal;
};

double trajLength(const StdVector<SE3> &traj) {
  CHECK_GE(traj.size(), 2);
  double len = 0;
  for (int i = 0; i < traj.size() - 1; ++i)
    len += (traj[i].inverse() * traj[i + 1]).translation().norm();
  return len;
}

TEST_F(SerializationTest, canBeInterrupted) {
  for (int frameInd = FLAGS_start;
       frameInd < FLAGS_start + FLAGS_count_before_interruption; ++frameInd) {
    auto frame = datasetReader->getFrame(frameInd);
    std::cout << "add frame #" << frameInd << std::endl;
    dsoOriginal->addFrame(frame, frameInd);
  }
  std::cout << "Interrupt: ";
  std::cout.flush();

  fs::path snapshotDir = outDir / "snapshot";
  dsoOriginal->saveSnapshot(snapshotDir);

  std::cout << "saved, ";
  std::cout.flush();

  SnapshotLoader snapshotLoader(datasetReader.get(), &cam, snapshotDir,
                                settings);
  TrajectoryWriter trajectoryWriterRestored(outDir, "oldstyle_restored.txt",
                                            FLAGS_traj_restored);
  Observers restoredObservers = createObservers(trajectoryWriterRestored);
  std::unique_ptr<DsoSystem> dsoRestored(
      new DsoSystem(snapshotLoader, restoredObservers, settings));

  std::cout << "loaded" << std::endl;

  for (int frameInd = FLAGS_start + FLAGS_count_before_interruption;
       frameInd < FLAGS_start + FLAGS_count; ++frameInd) {
    auto frame = datasetReader->getFrame(frameInd);
    std::cout << "add frame #" << frameInd << " [1/2 .. ";
    std::cout.flush();
    dsoOriginal->addFrame(frame, frameInd);
    std::cout << "2/2]" << std::endl;
    dsoRestored->addFrame(frame, frameInd);
  }

  dsoOriginal.reset();
  dsoRestored.reset();

  auto originalTraj = trajectoryWriterClean.writtenFrameToWorld();
  auto restoredTraj = trajectoryWriterRestored.writtenFrameToWorld();
  originalTraj.erase(originalTraj.begin(), originalTraj.begin() +
                                               originalTraj.size() -
                                               restoredTraj.size());
  double len = trajLength(originalTraj);
  SE3 lastOrig = originalTraj.back();
  SE3 lastRestored = restoredTraj.back();
  SE3 diff = lastOrig * lastRestored.inverse();
  double transErr = diff.translation().norm() / len;
  double rotErr = (180 / M_PI) * diff.so3().log().norm() / len;
  LOG(INFO) << "translational err = " << transErr * 100 << "%";
  LOG(INFO) << "rotational err = " << rotErr << "deg / m";
  EXPECT_LT(transErr, maxTransErr);
  EXPECT_LT(rotErr, maxRotErr);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  return RUN_ALL_TESTS();
}
