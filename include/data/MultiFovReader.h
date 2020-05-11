#ifndef INCLUDE_MULTIFOVREADER
#define INCLUDE_MULTIFOVREADER

#include "data/DatasetReader.h"
#include <iostream>
#include <opencv2/opencv.hpp>

namespace mdso {

class MultiFovReader : public DatasetReader {
public:
  class Depths : public FrameDepths {
  public:
    Depths(const cv::Mat1d &depths);

    std::optional<double> depth(int camInd, const Vec2 &point) const override;

  private:
    Eigen::AlignedBox2d bound;
    cv::Mat1d depths;
  };

  static bool isMultiFov(const fs::path &datasetDir);

  MultiFovReader(const fs::path &newDatasetDir);

  int numFrames() const override;
  std::vector<Timestamp> timestampsFromInd(int frameInd) const override;
  int firstTimestampToInd(Timestamp timestamp) const override;
  std::vector<FrameEntry> frame(int frameInd) const override;
  CameraBundle cam() const override;
  std::unique_ptr<FrameDepths> depths(int frameInd) const override;
  bool hasFrameToWorld(int frameInd) const override;
  SE3 frameToWorld(int frameInd) const override;

private:
  fs::path datasetDir;
  CameraBundle mCam;
  StdVector<SE3> frameToWorldGT;
};

} // namespace mdso

#endif
