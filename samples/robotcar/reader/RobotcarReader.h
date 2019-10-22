#ifndef INCLUDE_ROBOTCARREADER
#define INCLUDE_ROBOTCARREADER

#include "system/CameraBundle.h"
#include <filesystem>
#include <opencv2/opencv.hpp>

using namespace mdso;

struct ReaderSettings {
  Settings::CameraModel cam = {};
  bool fillVoGaps = false;
};

class RobotcarReader {
public:
  enum CamName { CAM_LEFT, CAM_REAR, CAM_RIGHT };

  struct FrameEntry {
    cv::Mat3b frame;
    Timestamp timestamp;
  };

  static constexpr int imageWidth = 1024, imageHeight = 1024;
  static constexpr int numCams = 3;

  RobotcarReader(const fs::path &_chunkDir, const fs::path &modelsDir,
                 const fs::path &extrinsicsDir,
                 const ReaderSettings &_settings = {});

  void provideMasks(const fs::path &masksDir);

  int numFrames() const;
  std::array<FrameEntry, numCams> frame(int idx) const;

  SE3 tsToTs(Timestamp src, Timestamp dst) const;

  std::vector<Vec3> getLmsFrontCloud(Timestamp from, Timestamp to,
                                     Timestamp base) const;
  std::vector<Vec3> getLmsRearCloud(Timestamp from, Timestamp to,
                                    Timestamp base) const;
  std::vector<Vec3> getLdmrsCloud(Timestamp from, Timestamp to,
                                  Timestamp base) const;

  std::array<StdVector<std::pair<Vec2, double>>, numCams>
  project(Timestamp from, Timestamp to, Timestamp base, bool useLmsFront = true,
          bool useLmsRear = true, bool useLdmrs = true);
  std::array<StdVector<std::pair<Vec2, double>>, numCams>
  project(const std::vector<Vec3> &cloud);

  inline CameraBundle &cam() { return mCam; }
  inline const CameraBundle &cam() const { return mCam; }
  inline const std::vector<Timestamp> &leftTs() const { return mLeftTs; }
  inline const std::vector<Timestamp> &rearTs() const { return mRearTs; }
  inline const std::vector<Timestamp> &rightTs() const { return mRightTs; }
  inline const std::vector<Timestamp> &camTs(CamName camName) const {
    return camName == CAM_LEFT ? leftTs()
                               : camName == CAM_REAR ? rearTs() : rightTs();
  }
  inline const std::vector<Timestamp> &lmsFrontTs() const {
    return mLmsFrontTs;
  }
  inline const std::vector<Timestamp> &lmsRearTs() const { return mLmsRearTs; }
  inline const std::vector<Timestamp> &ldmrsTs() const { return mLdmrsTs; }
  inline const std::vector<Timestamp> &voTs() const { return mVoTs; }

  inline const StdVector<SE3> &getVoBodyToFirst() const {
    return voBodyToFirst;
  }
  inline bool masksProvided() const { return mMasksProvided; }

private:
  static const SE3 camToImage;

  static CameraBundle createFromData(const fs::path &modelsDir,
                                     const SE3 &bodyToLeft,
                                     const SE3 &bodyToRear,
                                     const SE3 &bodyToRight, int w, int h,
                                     const Settings::CameraModel &camSettings);

  void syncTimestamps();
  SE3 tsToFirst(Timestamp ts) const;
  void getPointCloudHelper(std::vector<Vec3> &cloud, const fs::path &scanDir,
                           const SE3 &sensorToBody, Timestamp base,
                           const std::vector<Timestamp> &timestamps,
                           Timestamp from, Timestamp to, bool isLdmrs) const;

  SE3 bodyToLeft, bodyToRear, bodyToRight;
  SE3 bodyToLmsFront, bodyToLmsRear;
  SE3 bodyToLdmrs;
  StdVector<SE3> voBodyToFirst;
  CameraBundle mCam;
  fs::path chunkDir;
  fs::path leftDir, rearDir, rightDir;
  fs::path lmsFrontDir, lmsRearDir;
  fs::path ldmrsDir;
  std::vector<Timestamp> mLeftTs, mRearTs, mRightTs;
  std::vector<Timestamp> mLmsFrontTs, mLmsRearTs;
  std::vector<Timestamp> mLdmrsTs;
  std::vector<Timestamp> mVoTs;
  bool mMasksProvided;
  ReaderSettings settings;
};

#endif
