#ifndef INCLUDE_ROBOTCARREADER
#define INCLUDE_ROBOTCARREADER

#include "data/DatasetReader.h"
#include "util/Terrain.h"
#include <filesystem>
#include <opencv2/opencv.hpp>

using namespace mdso;

struct ReaderSettings {
  Settings::CameraModel cam = {};
  Settings::Triangulation triangulation = {};

  static constexpr bool default_fillVoGaps = false;
  bool fillVoGaps = default_fillVoGaps;

  static constexpr bool default_correctRtk = true;
  bool correctRtk = default_correctRtk;

  static constexpr double default_projectedTimeWindow = 2;
  double projectedTimeWindow = default_projectedTimeWindow;

  static constexpr int default_maxProjectedPoints = 20'000;
  int maxProjectedPoints = default_maxProjectedPoints;

  static constexpr double default_boxFilterSize = 0.2;
  double boxFilterSize = default_boxFilterSize;
};

class RobotcarReader : public DatasetReader {
public:
  static constexpr int imageWidth = 1024, imageHeight = 1024;
  static constexpr int numCams = 3;

  class Depths : public FrameDepths {
  public:
    Depths(const CameraBundle &cam,
           const std::array<StdVector<std::pair<Vec2, double>>, numCams>
               &projected,
           const Settings::Triangulation &settings = {});

    std::optional<double> depth(int camInd, const Vec2 &point) const override;
    cv::Mat3b draw(cv::Mat3b frames[]);

  private:
    std::array<Terrain, numCams> terrains;
  };

  RobotcarReader(const fs::path &_chunkDir, const fs::path &modelsDir,
                 const fs::path &extrinsicsDir,
                 const std::optional<fs::path> &rtkDir,
                 const ReaderSettings &_settings = {});
  void provideMasks(const fs::path &masksDir);

  int numFrames() const override;
  std::vector<FrameEntry> frame(int frameInd) const override;

  CameraBundle cam() const override { return mCam; }

  std::unique_ptr<FrameDepths> depths(int frameInd) const override;
  std::optional<SE3> frameToWorld(int frameInd) const override;

  SE3 tsToWorld(Timestamp ts, bool useVo = false) const;

  std::vector<Vec3> getLmsFrontCloud(Timestamp from, Timestamp to,
                                     Timestamp base) const;
  std::vector<Vec3> getLmsRearCloud(Timestamp from, Timestamp to,
                                    Timestamp base) const;
  std::vector<Vec3> getLdmrsCloud(Timestamp from, Timestamp to,
                                  Timestamp base) const;

  std::array<StdVector<std::pair<Vec2, double>>, numCams>
  project(Timestamp from, Timestamp to, Timestamp base, bool useLmsFront = true,
          bool useLmsRear = true, bool useLdmrs = true) const;
  std::array<StdVector<std::pair<Vec2, double>>, numCams>
  project(const std::vector<Vec3> &cloud) const;

  inline const std::vector<Timestamp> &lmsFrontTs() const {
    return mLmsFrontTs;
  }

  inline const StdVector<SE3> &getGtBodyToWorld() const {
    return gtBodyToWorld;
  }

  inline bool masksProvided() const { return mMasksProvided; }

  Timestamp minTs() const;
  Timestamp maxTs() const;

  Timestamp tsFromInd(int frameInd) const;
  int indFromTs(Timestamp ts) const;

private:
  static const SE3 camToImage;

  static CameraBundle createFromData(const fs::path &modelsDir,
                                     const SE3 &bodyToLeft,
                                     const SE3 &bodyToRear,
                                     const SE3 &bodyToRight, int w, int h,
                                     const Settings::CameraModel &camSettings);

  void syncTimestamps();
  void getPointCloudHelper(std::vector<Vec3> &cloud, const fs::path &scanDir,
                           const SE3 &sensorToBody, Timestamp base,
                           const std::vector<Timestamp> &timestamps,
                           Timestamp from, Timestamp to, bool isLdmrs) const;

  void printVoAndRtk() const;

  SE3 bodyToLeft, bodyToRear, bodyToRight;
  SE3 bodyToLmsFront, bodyToLmsRear;
  SE3 bodyToLdmrs;
  SE3 bodyToIns;
  StdVector<SE3> gtBodyToWorld;
  StdVector<SE3> voBodyToWorld;
  CameraBundle mCam;
  fs::path chunkDir;
  fs::path leftDir, rearDir, rightDir;
  fs::path lmsFrontDir, lmsRearDir;
  fs::path ldmrsDir;
  std::vector<Timestamp> mLeftTs, mRearTs, mRightTs;
  std::vector<Timestamp> mLmsFrontTs, mLmsRearTs;
  std::vector<Timestamp> mLdmrsTs;
  std::vector<Timestamp> mGroundTruthTs;
  std::vector<Timestamp> mVoTs;
  bool mMasksProvided;
  ReaderSettings settings;
};

#endif
