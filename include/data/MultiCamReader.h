#ifndef INCLUDE_MULTICAMREADER
#define INCLUDE_MULTICAMREADER

#include "data/DatasetReader.h"
#include "util/Terrain.h"

namespace mdso {

class MultiCamReader : public DatasetReader {
  static constexpr int mNumFrames = 3000;
  static constexpr int imgWidth = 640, imgHeight = 480;

public:
  struct Settings {
    Settings();

    static constexpr bool default_useInterpoatedDepths = true;
    bool useInterpolatedDepths = default_useInterpoatedDepths;

    static constexpr int default_numKeyPoints = 200;
    int numKeyPoints = default_numKeyPoints;

    static const std::vector<std::string> default_camNames;
    std::vector<std::string> camNames = default_camNames;

    int numCams() const;
  };

  class Depths : public FrameDepths {
  public:
    Depths(const fs::path &datasetDir, int frameInd,
           const Settings &newSettings);

    std::optional<double> depth(int camInd, const Vec2 &point) const override;

  private:
    std::vector<cv::Mat1f> depths;
    static const Eigen::AlignedBox2d boundingBox;

    Settings settings;
  };

  class InterpolatedDepths : public FrameDepths {
  public:
    InterpolatedDepths(const MultiCamReader *multiCamReader,
                       const CameraBundle *cam, int frameInd, int numFeatures,
                       const Settings &newSettings);

    std::optional<double> depth(int camInd, const Vec2 &point) const override;

  private:
    std::vector<Terrain> depths;
    Settings settings;
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
  static CameraBundle
  createCameraBundle(const fs::path &datasetDir,
                     const std::vector<std::string> &camNames);
  static StdVector<SE3> readBodyToWorld(const fs::path &datasetDir);

  Settings settings;
  fs::path datasetDir;
  CameraBundle mCam;
  StdVector<SE3> bodyToWorld;
};

} // namespace mdso

#endif
