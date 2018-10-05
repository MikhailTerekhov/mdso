#include <fstream>
#include "system/dsosystem.h"

using namespace fishdso;

int main(int argc, char **argv) {
  std::string usage = R"abacaba(Usage: track cam framesdir start count output outpred
Where cam names a file with camera calibration
framesdir names a directory with video frames.
Images from this directory should be named #.jpg
# stands for 9-symbol integer aligned with leading zeros
start is the number of the frame to start from
count is the number of frames for system to process
output stands for file to put calculated positions into
outpred stands for file to put prediction info it)abacaba";

  if (argc != 7) {
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

  for (int it = start; it < start + N; ++it) {
    char filename[256];
    sprintf(filename, "%s/%09d.jpg", argv[2], it);
    cv::Mat frame = cv::imread(filename);
    if (frame.data == NULL) {
      std::cerr << "frame named \"" << filename << "\" could not be read!"
                << std::endl;
      return 4;
    }
    std::cout << "put frame " << it << std::endl;
    dsoSystem.addFrame(frame);
  }
  
  std::ofstream output(argv[5]);
  dsoSystem.printTrackingInfo(output);
  
  std::ofstream outpred(argv[6]); 
  dsoSystem.printPredictionInfo(outpred);
  
  std::ofstream outPly("points.ply");
  dsoSystem.printLastKfInPly(outPly);

  return 0;
}
