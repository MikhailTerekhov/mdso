#ifndef INCLUDE_MULTICAMREADER
#define INCLUDE_MULTICAMREADER

#include "data/DatasetReader.h"
#include "util/Terrain.h"

namespace mdso {

class MultiCamReader : public DatasetReader {
  static constexpr int mNumFrames = 3000;
  static constexpr int numCams = 4;
  static constexpr int imgWidth = 640, imgHeight = 480;
  static const std::string camNames[numCams];

public:
  struct Settings {
    Settings();

    static constexpr bool default_useInterpoatedDepths = true;
    bool useInterpolatedDepths = default_useInterpoatedDepths;

    static constexpr int default_numKeyPoints = 200;
    int numKeyPoints = default_numKeyPoints;
  };

  class Depths : public FrameDepths {
  public:
    Depths(const fs::path &datasetDir, int frameInd);

    std::optional<double> depth(int camInd, const Vec2 &point) const override;

  private:
    std::array<cv::Mat1f, numCams> depths;
    Eigen::AlignedBox2d boundingBox;
  };

  class InterpolatedDepths : public FrameDepths {
  public:
    InterpolatedDepths(const MultiCamReader *multiCamReader,
                       const CameraBundle *cam, int frameInd, int numFeatures);

    std::optional<double> depth(int camInd, const Vec2 &point) const override;

  private:
    std::vector<Terrain> depths;
  };

  static bool isMultiCam(const fs::path &datasetDir);

  MultiCamReader(const fs::path &datasetDir,
                 const MultiCamReader::Settings &settings = {});

  int numFrames() const override;
  int firstTimestampToInd(Timestamp timestamp) const override;
  std::vector<Timestamp> timestampsFromInd(int frameInd) const override;
  std::vector<FrameEntry> frame(int frameInd) const override;
  CameraBundle cam() const override;
  std::unique_ptr<FrameDepths> depths(int frameInd) const override;
  bool hasFrameToWorld(int frameInd) const override;
  SE3 frameToWorld(int frameInd) const override;

  std::unique_ptr<Depths> groundTruthDepths(int frameInd) const;
  std::unique_ptr<InterpolatedDepths> interpolatedDepths(int frameInd) const;

private:
  static CameraBundle createCameraBundle(const fs::path &datasetDir);
  static StdVector<SE3> readBodyToWorld(const fs::path &datasetDir);

  fs::path datasetDir;
  CameraBundle mCam;
  StdVector<SE3> bodyToWorld;
  Settings settings;
};

} // namespace mdso

#endif
