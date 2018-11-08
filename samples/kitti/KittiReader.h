#include "system/CameraModel.h"
#include "util/types.h"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace fishdso;

class KittiReader {
public:
  KittiReader(const std::string &newKittiDir, int sequenceNum, int startFrame);

  cv::Mat getFrame(int globalFrameNum);
  SE3 getWorldToFrameGT(int globalFrameNum);

  std::unique_ptr<CameraModel> cam;

private:
  std::string kittiDir;
  int sequenceNum;
  Mat33 K;
  StdVector<SE3> worldToFrameGT;
};
