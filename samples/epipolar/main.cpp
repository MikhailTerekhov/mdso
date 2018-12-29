#include "system/DelaunayDsoInitializer.h"
#include "system/ImmaturePoint.h"
#include "util/defs.h"
#include <opencv2/opencv.hpp>

using namespace fishdso;

DEFINE_int32(x, 100, "x-coordinate of the point on base frame");
DEFINE_int32(y, 100, "y-coordinate of the point on base frame");

int main(int argc, char **argv) {
  std::string usage = R"abacaba(Usage: "stereo cam img1 img2"
Where cam names a file with camera calibration;
img1 and img2 name files with two frames to track.)abacaba";

  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::SetUsageMessage(usage);
  google::InitGoogleLogging(argv[0]);

  if (argc != 4) {
    std::cout << "Wrong number of arguments!\n" << usage << std::endl;
    return 0;
  }

  CameraModel cam(1920, 1208, argv[1]);
  DelaunayDsoInitializer initializer(nullptr, &cam, DelaunayDsoInitializer::NO_DEBUG);
  cv::Mat frame1, frame2;
  frame1 = cv::imread(argv[2]);
  if (frame1.data == NULL) {
    std::cout << "img1 could not be found or read!" << std::endl;
    return 0;
  }

  frame2 = cv::imread(argv[3]);
  if (frame2.data == NULL) {
    std::cout << "img2 could not be found or read!" << std::endl;
    return 0;
  }

  FLAGS_first_frames_skip = 0;
  FLAGS_perform_full_tracing = true;
  initializer.addFrame(frame1, 1);
  initializer.addFrame(frame2, 2);
  std::vector<KeyFrame> keyFrames =
      initializer.createKeyFrames();
  keyFrames[0].deactivateAllOptimized();

  for (auto &ip : keyFrames[0].immaturePoints)
    ip->traceOn(*keyFrames[1].preKeyFrame, ImmaturePoint::DRAW_EPIPOLE);


  return 0;
}
