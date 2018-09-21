#include "system/dsosystem.h"

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
  DsoSystem sys(&cam);
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

  //  cv::Mat ds = boxFilterPyrUp<cv::Vec3b>(frame1);
  //  cv::imshow("original", frame1);
  //  cv::imshow("downsampled", ds);

  //  cv::waitKey();
  //  return 0;

  sys.addFrame(frame1);
  sys.addFrame(frame2);
}
