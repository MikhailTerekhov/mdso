#include "KittiReader.h"
#include "system/DsoSystem.h"
#include "util/defs.h"
#include "util/util.h"
#include <gflags/gflags.h>

int main(int argc, char **argv) {
  std::string usage =
      R"abacaba(Usage: kitti kitti_dir start count
Where kitti_dir names a directory with KITTI dataset.
It should contain "\"sequences\" subdirectory.
start is the number of the frame to start from
count is the number of frames for system to process)abacaba";

  gflags::ParseCommandLineFlags(&argc, &argv, true);
  gflags::SetUsageMessage(usage);
  google::InitGoogleLogging(argv[0]);

  if (argc != 4) {
    std::cerr << "Wrong number of arguments!\n" << usage << std::endl;
    return 1;
  }

  int start = 0;
  if (sscanf(argv[2], "%d", &start) != 1) {
    std::cerr << "starting frame could not be read!\n" << usage << std::endl;
    return 2;
  }

  int N = 0;
  if (sscanf(argv[3], "%d", &N) != 1) {
    std::cerr << "Number of frames could not be read!\n" << usage << std::endl;
    return 3;
  }

  KittiReader reader(argv[1], 1, start);
  DsoSystem dsoSystem(reader.cam.get());

  for (int it = start; it < start + N; ++it) {
    cv::Mat frame = reader.getFrame(it);
    // if (it == start) {
      // std::cout << "img channels = " << frame.channels() << std::endl;
      // cv::Mat fc = frame.clone();
      // putDot(fc, toCvPoint(reader.cam->getImgCenter()), CV_RED);
      // cv::imshow("centered", fc);
      // cv::waitKey();
    // }
    std::cout << "put frame #" << it << std::endl;
    dsoSystem.addGroundTruthPose(it, reader.getWorldToFrameGT(it));
    dsoSystem.addFrame(frame, it);
  }

  return 0;
}
