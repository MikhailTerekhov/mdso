#ifndef INCLUDE_MULTICAMREADER
#define INCLUDE_MULTICAMREADER

#include "data/DatasetReader.h"

namespace mdso {

class MultiCamReader : public DatasetReader {
  static constexpr int mNumFrames = 3000;
  static constexpr int numCams = 4;
  static constexpr int imgWidth = 640, imgHeight = 480;
  static const std::string camNames[numCams];

public:
  class Depths : public FrameDepths {
  public:
    Depths(const fs::path &datasetDir, int frameInd);

    std::optional<double> depth(int camInd, const Vec2 &point) const override;

  private:
    std::array<cv::Mat1f, numCams> depths;
    Eigen::AlignedBox2d boundingBox;
  };

  static bool isMultiCam(const fs::path &datasetDir);

  MultiCamReader(const fs::path &datasetDir);

  int numFrames() const override;
  int firstTimestampToInd(Timestamp timestamp) const override;
  std::vector<Timestamp> timestampsFromInd(int frameInd) const override;
  std::vector<FrameEntry> frame(int frameInd) const override;
  CameraBundle cam() const override;
  std::unique_ptr<FrameDepths> depths(int frameInd) const override;
  bool hasFrameToWorld(int frameInd) const override;
  SE3 frameToWorld(int frameInd) const override;

private:
  static CameraBundle createCameraBundle(const fs::path &datasetDir);
  static StdVector<SE3> readBodyToWorld(const fs::path &datasetDir);

  fs::path datasetDir;
  CameraBundle mCam;
  StdVector<SE3> bodyToWorld;
};

} // namespace mdso

#endif
