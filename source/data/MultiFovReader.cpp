#include "data/MultiFovReader.h"
#include "internal/data/getMfovCam.h"
#include "util/types.h"

namespace mdso {

MultiFovReader::Depths::Depths(const cv::Mat1d &depths)
    : bound(Vec2::Zero(), Vec2(depths.cols - 1, depths.rows - 1))
    , depths(depths) {}

std::optional<double> MultiFovReader::Depths::depth(int camInd,
                                                    const Vec2 &point) const {
  CHECK_EQ(camInd, 0);
  if (bound.contains(point))
    return std::make_optional(depths(toCvPoint(point)));
  else
    return std::nullopt;
}

bool MultiFovReader::isMultiFov(const fs::path &datasetDir) {
  fs::path infoDir = datasetDir / "info";
  fs::path dataDir = datasetDir / "data";
  return fs::exists(infoDir / "intrinsics.txt") &&
         fs::exists(infoDir / "groundtruth.txt") &&
         fs::exists(dataDir / "img") && fs::exists(dataDir / "depth");
}

CameraBundle getCam(const fs::path &datasetDir) {
  CameraModel cam = getMfovCam(datasetDir / "info" / "intrinsics.txt");
  SE3 id;
  return CameraBundle(&id, &cam, 1);
}

MultiFovReader::MultiFovReader(const fs::path &newMultiFovDir)
    : datasetDir(newMultiFovDir)
    , mCam(getCam(newMultiFovDir)) {
  fs::path posesFName = datasetDir / fs::path("info/groundtruth.txt");
  std::ifstream posesIfs(posesFName);
  CHECK(posesIfs.is_open()) << "could not open ground truth poses file \""
                            << posesFName.native() << "\"";
  frameToWorldGT.push_back(SE3());
  while (!posesIfs.eof()) {
    int frameNum;
    Vec3 trans;
    Eigen::Quaterniond quat;
    posesIfs >> frameNum >> trans[0] >> trans[1] >> trans[2] >> quat.x() >>
        quat.y() >> quat.z() >> quat.w();
    frameToWorldGT.push_back(SE3(quat, trans));
  }
}

int MultiFovReader::numFrames() const { return frameToWorldGT.size(); }

int MultiFovReader::firstTimestampToInd(Timestamp timestamp) const {
  if (timestamp < 0)
    return 0;
  if (timestamp > numFrames())
    return numFrames();

  return int(timestamp);
}

std::vector<Timestamp> MultiFovReader::timestampsFromInd(int frameInd) const {
  CHECK_GE(frameInd, 0);
  CHECK_LT(frameInd, numFrames());

  return std::vector<Timestamp>(1, frameInd);
}

std::vector<DatasetReader::FrameEntry>
MultiFovReader::frame(int frameInd) const {
  CHECK_GE(frameInd, 0);
  CHECK_LT(frameInd, numFrames());

  char innerName[15];
  snprintf(innerName, 15, "img%04i_0.png", frameInd);
  fs::path frameFName = datasetDir / fs::path("data/img") / fs::path(innerName);
  std::vector<FrameEntry> frame(1);
  frame[0].frame = cv::imread(frameFName.native());
  frame[0].timestamp = frameInd;
  CHECK_NOTNULL(frame[0].frame.data);
  return frame;
}

CameraBundle MultiFovReader::cam() const { return mCam; }

std::unique_ptr<FrameDepths> MultiFovReader::depths(int frameInd) const {
  CHECK_GE(frameInd, 0);
  CHECK_LT(frameInd, numFrames());

  char innerName[18];
  snprintf(innerName, 18, "img%04i_0.depth", frameInd);
  fs::path depthsFName =
      datasetDir / fs::path("data/depth") / fs::path(innerName);
  std::ifstream depthsIfs(depthsFName);
  CHECK(depthsIfs.is_open())
      << "could not open depths file \"" << depthsFName.native() << "\"";
  cv::Mat1d depths(mCam.bundle[0].cam.getHeight(),
                   mCam.bundle[0].cam.getWidth());
  for (int y = 0; y < depths.rows; ++y)
    for (int x = 0; x < depths.cols; ++x)
      depthsIfs >> depths(y, x);

  return std::unique_ptr<FrameDepths>(new Depths(depths));
}

bool MultiFovReader::hasFrameToWorld(int frameInd) const {
  return frameInd >= 0 && frameInd < numFrames();
}

SE3 MultiFovReader::frameToWorld(int frameInd) const {
  CHECK_GE(frameInd, 0);
  CHECK_LT(frameInd, numFrames());
  return frameToWorldGT[frameInd];
}

} // namespace mdso
