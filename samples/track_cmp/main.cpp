#include "system/DsoSystem.h"
#include <fstream>
#include <gflags/gflags.h>
#include <glog/logging.h>

using namespace fishdso;

int main(int argc, char **argv) {
  std::string usage =
      R"abacaba(Usage: track_cmp cam framesdir start count
Where cam names a file with camera calibration
framesdir names a directory with video frames.
Images from this directory should be named #.jpg
# stands for 9-symbol integer aligned with leading zeros
start is the number of the frame to start from
count is the number of frames for system to process)abacaba";

  gflags::ParseCommandLineFlags(&argc, &argv, true);
  gflags::SetUsageMessage(usage);
  google::InitGoogleLogging(argv[0]);

  if (argc != 5) {
    std::cerr << "Wrong number of arguments!\n" << usage << std::endl;
    return 1;
  }

  CameraModel cam(1920, 1208, argv[1]);
  DsoSystem dsoSystem(&cam);

  int start = 0;
  if (sscanf(argv[3], "%d", &start) != 1) {
    std::cerr << "starting frame could not be read!\n" << usage << std::endl;
    return 2;
  }

  int N = 0;
  if (sscanf(argv[4], "%d", &N) != 1) {
    std::cerr << "Number of frames could not be read!\n" << usage << std::endl;
    return 3;
  }

  cv::Mat frame;
  for (int it = start; it < start + N; ++it) {
    char filename[256];
    sprintf(filename, "%s/%09d.jpg", argv[2], it);
    frame = cv::imread(filename);
    if (frame.data == NULL) {
      std::cerr << "frame named \"" << filename << "\" could not be read!"
                << std::endl;
      return 4;
    }
    std::cout << "put frame " << it << std::endl;
    dsoSystem.addFrame(frame);
  }

  return 0;
}
