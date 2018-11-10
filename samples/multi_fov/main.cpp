#include "MultiFovReader.h"
#include "system/DsoSystem.h"
#include <iostream>

using namespace fishdso;

int main(int argc, char **argv) {
  std::string usage =
      R"abacaba(Usage: multi_fov data_dir
Where data_dir names a directory with MultiFoV fishseye dataset.
It should contain "info" and "data" subdirectories.)abacaba";

  if (argc != 2) {
    std::cerr << "Wrong number of arguments!\n" << usage << std::endl;
    return 1;
  }
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  gflags::SetUsageMessage(usage);
  google::InitGoogleLogging(argv[0]);

  MultiFovReader reader(argv[1]);

  Mat33 K;
  // clang-format off
  K << 100,   0, 320,
         0, 100, 240,
         0,   0,   1;
  // clang-format on
  cv::Mat frame = reader.getFrame(1);
  cv::Mat undistorted = reader.cam->undistort<cv::Vec3b>(frame, K);
  cv::imshow("orig", frame);
  cv::imshow("undistort", undistorted);
  cv::waitKey();

  return 0;
}
