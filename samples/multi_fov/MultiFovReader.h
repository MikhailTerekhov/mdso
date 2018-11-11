#include "system/CameraModel.h"
#include "util/types.h"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace fishdso;

class MultiFovReader {
public:
  MultiFovReader(const std::string &newDatasetDir);

  cv::Mat getFrame(int globalFrameNum) const;
  cv::Mat1f getDepths(int globalFrameNum) const;
  SE3 getWorldToFrameGT(int globalFrameNum) const;

  std::unique_ptr<CameraModel> cam;

private:
  std::string datasetDir;
  StdVector<SE3> worldToFrameGT;
};
