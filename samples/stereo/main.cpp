#include "system/DelaunayDsoInitializer.h"

#include <string>

using namespace fishdso;

int main(int argc, char **argv) {
  std::string usage = R"abacaba(Usage: "stereo cam img1 img2"
Where cam names a file with camera calibration;
img1 and img2 name files with two frames to track.)abacaba";

  if (argc != 4) {
    std::cout << "Wrong number of arguments!\n" << usage << std::endl;
    return 0;
  }

  CameraModel cam(1920, 1208, argv[1]);
  DelaunayDsoInitializer initializer(nullptr, &cam, DelaunayDsoInitializer::SPARSE_DEPTHS);
  cv::Mat frame1, frame2;
  frame1 = cv::imread(argv[2]);
  if (frame1.data == NULL) {
    std::cout << "img1 could not be found or read!" << std::endl;
    return 0;
  }

  // cv::Mat res;
  // Mat33 K = (Mat33()
  // <<  1866.0 , 0.0    , 960.0
  // , 0.0    , 1866.0 , 960.0
  // , 0.0    , 0.0    , 1.0).finished();
  // cam.undistort<cv::Vec3b>(frame1, res, K);

  frame2 = cv::imread(argv[3]);
  if (frame2.data == NULL) {
    std::cout << "img2 could not be found or read!" << std::endl;
    return 0;
  }

  FLAGS_first_frames_skip = 0;
  initializer.addFrame(frame1, 1);
  initializer.addFrame(frame2, 2);
  initializer.createKeyFrames();
}
