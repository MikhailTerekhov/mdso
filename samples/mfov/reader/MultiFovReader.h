#ifndef INCLUDE_MULTIFOVREADER
#define INCLUDE_MULTIFOVREADER

#include "system/CameraModel.h"
#include "util/types.h"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace fishdso;

class MultiFovReader {
public:
  MultiFovReader(const fs::path &newDatasetDir);

  cv::Mat getFrame(int globalFrameNum) const;
  cv::Mat1d getDepths(int globalFrameNum) const;
  SE3 getWorldToFrameGT(int globalFrameNum) const;
  const StdVector<SE3> &getAllWorldToFrameGT() const;
  int getFrameCount() const;

  std::unique_ptr<CameraModel> cam;

private:
  fs::path datasetDir;
  StdVector<SE3> worldToFrameGT;
};

#endif
