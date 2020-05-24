#include "data/MultiCamReader.h"
#include "internal/data/getMfovCam.h"

namespace mdso {

const std::string MultiCamReader::camNames[] = {"front", "right", "rear",
                                                "left"};

cv::Mat1f readBinMat(const fs::path &fname, int imgWidth, int imgHeight) {
  std::ifstream depthsIfs(fname, std::ios::binary);
  CHECK(depthsIfs.is_open()) << "failed to open file: " << fname;
  cv::Mat1f depths(imgHeight, imgWidth);
  for (int y = 0; y < imgHeight; ++y)
    for (int x = 0; x < imgWidth; ++x)
      depthsIfs.read(reinterpret_cast<char *>(&depths(y, x)), sizeof(float));
  return depths;
}

MultiCamReader::Depths::Depths(const fs::path &datasetDir, int frameInd)
    : boundingBox(Vec2::Zero(), Vec2(imgWidth - 1, imgHeight - 1)) {
  CHECK_GE(frameInd, 0);
  CHECK_LT(frameInd, mNumFrames);

  for (int ci = 0; ci < numCams; ++ci) {
    char innerName[15];
    snprintf(innerName, 15, "%s_%04d.bin", camNames[ci].c_str(), frameInd);
    fs::path depthPath =
        datasetDir / "data" / "depth" / camNames[ci] / innerName;
    depths[ci] = readBinMat(depthPath, imgWidth, imgHeight);
  }

  //  std::vector<float> depthVec;
  //  for (int ci = 0; ci < numCams; ++ci)
  //    for (int y = 0; y < imgHeight; ++y)
  //      for (int x = 0; x < imgWidth; ++x)
  //        depthVec.push_back(depths[ci](y, x));
  //
  //  std::sort(depthVec.begin(), depthVec.end());
  //  double minD = depthVec[0], maxD = depthVec[0.7 * depthVec.size()];
  //
  //  MultiCamReader r(datasetDir);
  //  auto f = r.frame(frameInd);
  //  for (int ci = 0; ci < numCams; ++ci) {
  //    cv::Mat3b drawn =
  //        drawDepthedFrame(cvtBgrToGray(f[ci].frame), depths[ci], minD, maxD);
  //    cv::imwrite("depths_" + camNames[ci] + "_" + std::to_string(frameInd) +
  //                    ".jpg",
  //                drawn);
  //  }
}

std::optional<double> MultiCamReader::Depths::depth(int camInd,
                                                    const Vec2 &point) const {
  if (!boundingBox.contains(point))
    return std::nullopt;
  CHECK_GE(camInd, 0);
  CHECK_LT(camInd, numCams);
  return std::optional<double>(double(depths[camInd](toCvPoint(point))));
}

bool MultiCamReader::isMultiCam(const fs::path &datasetDir) {
  fs::path infoDir = datasetDir / "info";
  fs::path intrinsicsDir = infoDir / "intrinsics";
  fs::path extrinsicsDir = infoDir / "extrinsics";
  fs::path dataDir = datasetDir / "data";
  fs::path imgDir = dataDir / "img";
  fs::path depthDir = dataDir / "depth";
  bool isMcam = fs::exists(infoDir / "body_to_world.txt");
  for (const auto &camName : camNames) {
    isMcam = isMcam && fs::exists(intrinsicsDir / (camName + ".txt"));
    isMcam = isMcam && fs::exists(extrinsicsDir / (camName + "_to_body.txt"));
    isMcam = isMcam && fs::exists(imgDir / camName);
    isMcam = isMcam && fs::exists(depthDir / camName);
  }
  return isMcam;
}

SE3 readFromMatrix3x4(std::istream &istream) {
  Mat34 mat;
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 4; ++c)
      istream >> mat(r, c);

  Mat33 R = mat.leftCols<3>();
  double Rdet = R.determinant();
  CHECK_NEAR(Rdet, 1, 1e-6);
  Eigen::JacobiSVD Rsvd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Mat33 U = Rsvd.matrixU(), V = Rsvd.matrixV();
  Mat33 Rfixed = U * V.transpose();

  return SE3(Rfixed, mat.col(3));
}

CameraBundle MultiCamReader::createCameraBundle(const fs::path &datasetDir) {
  fs::path extrinsicsDir(datasetDir / "info" / "extrinsics");
  fs::path intrinsicsDir(datasetDir / "info" / "intrinsics");

  StdVector<CameraModel> cams;
  StdVector<SE3> bodyToCam;
  cams.reserve(numCams);
  bodyToCam.reserve(numCams);
  for (const auto &camName : camNames) {
    std::ifstream extrinsicsFile(extrinsicsDir / (camName + "_to_body.txt"));
    CHECK(extrinsicsFile.is_open());
    bodyToCam.push_back(readFromMatrix3x4(extrinsicsFile).inverse());
    cams.push_back(getMfovCam(intrinsicsDir / (camName + ".txt")));
  }

  return CameraBundle(bodyToCam.data(), cams.data(), numCams);
}

StdVector<SE3> MultiCamReader::readBodyToWorld(const fs::path &datasetDir) {
  std::ifstream trajIfs(datasetDir / "info" / "body_to_world.txt");
  StdVector<SE3> bodyToWorld;
  bodyToWorld.reserve(mNumFrames);
  for (int fi = 0; fi < mNumFrames; ++fi)
    bodyToWorld.push_back(readFromMatrix3x4(trajIfs));
  return bodyToWorld;
}

MultiCamReader::MultiCamReader(const fs::path &_datasetDir)
    : datasetDir(_datasetDir)
    , mCam(createCameraBundle(_datasetDir))
    , bodyToWorld(readBodyToWorld(_datasetDir)) {}

int MultiCamReader::numFrames() const { return mNumFrames; }

int MultiCamReader::firstTimestampToInd(Timestamp timestamp) const {
  CHECK_GE(timestamp, 0);
  CHECK_LT(timestamp, mNumFrames);
  return int(timestamp);
}

std::vector<Timestamp> MultiCamReader::timestampsFromInd(int frameInd) const {
  if (frameInd < 0)
    return std::vector<Timestamp>(numCams, 0);
  if (frameInd >= mNumFrames)
    return std::vector<Timestamp>(numCams, mNumFrames - 1);
  return std::vector<Timestamp>(numCams, frameInd);
}

std::vector<DatasetReader::FrameEntry>
MultiCamReader::frame(int frameInd) const {
  CHECK_GE(frameInd, 0);
  CHECK_LT(frameInd, mNumFrames);
  std::vector<DatasetReader::FrameEntry> frame;
  frame.reserve(numCams);
  for (int ci = 0; ci < numCams; ++ci) {
    char innerName[15];
    snprintf(innerName, 15, "%s_%04d.jpg", camNames[ci].c_str(), frameInd);
    fs::path imagePath = datasetDir / "data" / "img" / camNames[ci] / innerName;
    cv::Mat3b image = cv::imread(imagePath.string());
    CHECK_NOTNULL(image.data);
    frame.push_back({image, frameInd});
  }
  return frame;
}

CameraBundle MultiCamReader::cam() const { return mCam; }

std::unique_ptr<FrameDepths> MultiCamReader::depths(int frameInd) const {
  return std::unique_ptr<FrameDepths>(new Depths(datasetDir, frameInd));
}

bool MultiCamReader::hasFrameToWorld(int frameInd) const {
  return frameInd >= 0 && frameInd < mNumFrames;
}

SE3 MultiCamReader::frameToWorld(int frameInd) const {
  CHECK_GE(frameInd, 0);
  CHECK_LT(frameInd, bodyToWorld.size());
  return bodyToWorld[frameInd];
}

} // namespace mdso