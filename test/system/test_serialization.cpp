#include "data/MultiFovReader.h"
#include "output/TrajectoryWriterDso.h"
#include "system/DsoSystem.h"
#include "system/IdentityPreprocessor.h"
#include "system/serialization.h"
#include "util/flags.h"
#include <gtest/gtest.h>

using namespace mdso;

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

Observers createObservers(TrajectoryWriterDso &trajectoryWriter) {
  Observers observers;
  observers.dso.push_back(&trajectoryWriter);
  return observers;
}

Settings getSettings() {
  Settings settings = getFlaggedSettings();
  settings.setKeyFrameDist(5);
  settings.optimization.useSelfWrittenOptimization = false;
  return settings;
}

class SerializationTest : public ::testing::Test {
public:
  static constexpr double maxTransErr = 1e-2;
  static constexpr double maxRotErr = 0.1;

  SerializationTest()
      : outDir(fs::path("output") / curTimeBrief())
      , datasetReader(new MultiFovReader(FLAGS_mfov_dir))
      , cam(datasetReader->cam())
      , identityPreprocessor()
      , settings(getSettings())
      , trajectoryWriterClean(outDir / FLAGS_traj_original)
      , originalObservers(createObservers(trajectoryWriterClean))
      , dsoOriginal(new DsoSystem(&cam, &identityPreprocessor,
                                  originalObservers, settings)) {
    fs::create_directories(outDir);
  }

protected:
  fs::path outDir;
  std::unique_ptr<DatasetReader> datasetReader;
  CameraBundle cam;
  IdentityPreprocessor identityPreprocessor;
  Settings settings;
  TrajectoryWriterDso trajectoryWriterClean;
  Observers originalObservers;
  std::unique_ptr<DsoSystem> dsoOriginal;
};

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

void saveTraj(const fs::path &fname, TrajectoryHolder *trajectoryHolder) {
  std::ofstream ofs(fname);
  int size = trajectoryHolder->trajectorySize();
  for (int i = 0; i < size; ++i)
    putInMatrixForm(ofs, trajectoryHolder->bodyToWorld(i));
}

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
    auto frame = cvtFrame(datasetReader->frame(frameInd));
    std::cout << "add frame #" << frameInd << std::endl;
    dsoOriginal->addMultiFrame(frame.first.data(), frame.second.data());
  }
  std::cout << "Interrupt: ";
  std::cout.flush();

  fs::path snapshotDir = outDir / "snapshot";
  dsoOriginal->saveSnapshot(snapshotDir);

  std::cout << "saved, ";
  std::cout.flush();

  SnapshotLoader snapshotLoader(datasetReader.get(), &identityPreprocessor,
                                &cam, snapshotDir, settings);
  auto restoredKeyFrames = snapshotLoader.load();
  TrajectoryWriterDso trajectoryWriterRestored(outDir / FLAGS_traj_restored);
  Observers restoredObservers = createObservers(trajectoryWriterRestored);
  std::unique_ptr<DsoSystem> dsoRestored(
      new DsoSystem(restoredKeyFrames, &cam, &identityPreprocessor,
                    restoredObservers, settings));

  std::cout << "loaded" << std::endl;

  for (int frameInd = FLAGS_start + FLAGS_count_before_interruption;
       frameInd < FLAGS_start + FLAGS_count; ++frameInd) {
    auto frame = cvtFrame(datasetReader->frame(frameInd));
    std::cout << "add frame #" << frameInd << " [1/2 .. ";
    std::cout.flush();
    dsoOriginal->addMultiFrame(frame.first.data(), frame.second.data());
    std::cout << "2/2]" << std::endl;
    dsoRestored->addMultiFrame(frame.first.data(), frame.second.data());
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
