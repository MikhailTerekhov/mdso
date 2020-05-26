#include "data/MultiCamReader.h"
#include "internal/data/getMfovCam.h"
#include <opencv2/features2d.hpp>

namespace mdso {

const std::string MultiCamReader::camNames[] = {"front", "right", "rear",
                                                "left"};

MultiCamReader::Settings::Settings() = default;

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
}

MultiCamReader::InterpolatedDepths::InterpolatedDepths(
    const MultiCamReader *multiCamReader, const CameraBundle *cam, int frameInd,
    int numFeatures) {
  auto orb = cv::ORB::create(numFeatures);
  auto correctDepths = multiCamReader->groundTruthDepths(frameInd);
  auto frame = multiCamReader->frame(frameInd);
  depths.reserve(numCams);
  for (int ci = 0; ci < numCams; ++ci) {
    std::vector<cv::KeyPoint> keyPoints;
    orb->detect(frame[ci].frame, keyPoints);
    StdVector<Vec2> points;
    std::vector<double> pointDepths;
    points.reserve(keyPoints.size());
    depths.reserve(keyPoints.size());
    for (const auto &kp : keyPoints) {
      Vec2 p = toVec2(kp.pt);
      auto d = correctDepths->depth(ci, p);
      if (!d)
        LOG(WARNING) << "ORB outside of an image: " << p.transpose();
      if (d && *d < 1e5) {
        points.push_back(p);
        pointDepths.push_back(*d);
      }
    }
    mdso::Settings::Triangulation triSettings;
    //    triSettings.epsSamePoints = 1e-5;
    depths.emplace_back(&cam->bundle[ci].cam, points, pointDepths, triSettings);

    cv::Mat3b img;
    img = frame[ci].frame.clone();
    depths.back().draw(img, CV_GREEN);
    cv::imwrite("terrain_" + std::to_string(ci) + ".jpg", img);
  }
}

std::optional<double>
MultiCamReader::InterpolatedDepths::depth(int camInd, const Vec2 &point) const {
  CHECK_GE(camInd, 0);
  CHECK_LT(camInd, numCams);
  double depth = 0;
  bool hasDepth = depths[camInd](point, depth);
  return hasDepth ? std::make_optional(depth) : std::nullopt;
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

MultiCamReader::MultiCamReader(const fs::path &_datasetDir,
                               const Settings &settings)
    : datasetDir(_datasetDir)
    , mCam(createCameraBundle(_datasetDir))
    , bodyToWorld(readBodyToWorld(_datasetDir))
    , settings(settings) {}

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

std::unique_ptr<MultiCamReader::InterpolatedDepths>
MultiCamReader::interpolatedDepths(int frameInd) const {
  return std::unique_ptr<InterpolatedDepths>(
      new InterpolatedDepths(this, &mCam, frameInd, settings.numKeyPoints));
}

std::unique_ptr<MultiCamReader::Depths>
MultiCamReader::groundTruthDepths(int frameInd) const {
  return std::unique_ptr<Depths>(new Depths(datasetDir, frameInd));
}

std::unique_ptr<FrameDepths> MultiCamReader::depths(int frameInd) const {
  if (settings.useInterpolatedDepths)
    return interpolatedDepths(frameInd);
  else
    return groundTruthDepths(frameInd);
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