#include "RobotcarReader.h"

#include <sophus/so3.hpp>

// clang-format off
const SE3 RobotcarReader::camToImage = 
    SE3((Mat44() <<
        0, 0, 1, 0,
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 0, 1).finished())
      .inverse();
// clang-format on

SE3 fromXyzrpy(double x, double y, double z, double roll, double pitch,
               double yaw) {
  SO3 rot = SO3::rotZ(yaw) * SO3::rotY(pitch) * SO3::rotX(roll);
  return SE3(rot, Vec3(x, y, z));
}

SE3 readFromExtrin(const fs::path &extrinsicsFile) {
  std::ifstream ifs(extrinsicsFile);
  CHECK(ifs.is_open()) << "Could not open extrinsics file " << extrinsicsFile;
  double x, y, z, roll, pitch, yaw;
  ifs >> x >> y >> z >> roll >> pitch >> yaw;
  SE3 result = fromXyzrpy(x, y, z, roll, pitch, yaw);
  return result;
}

SE3 readBodyToCam(const fs::path &extrinsicsFile) {
  SE3 result = readFromExtrin(extrinsicsFile);
  LOG(INFO) << "file: " << extrinsicsFile << "\nbody -> this:\n"
            << result.matrix();
  return result;
}

SE3 readBodyToLidar(const fs::path &extrinsicsFile) {
  SE3 lidarToBody = readFromExtrin(extrinsicsFile);
  LOG(INFO) << "file: " << extrinsicsFile << "\nthis -> body:\n"
            << lidarToBody.matrix();
  return lidarToBody.inverse();
}

CameraBundle
RobotcarReader::createFromData(const fs::path &modelsDir, const SE3 &bodyToLeft,
                               const SE3 &bodyToRear, const SE3 &bodyToRight,
                               int w, int h,
                               const Settings::CameraModel &camSettings) {
  fs::path leftModel = modelsDir / "mono_left.txt";
  fs::path rearModel = modelsDir / "mono_rear.txt";
  fs::path rightModel = modelsDir / "mono_right.txt";
  CameraModel models[RobotcarReader::numCams] = {
      CameraModel(w, h, leftModel, CameraModel::POLY_MAP, camSettings),
      CameraModel(w, h, rearModel, CameraModel::POLY_MAP, camSettings),
      CameraModel(w, h, rightModel, CameraModel::POLY_MAP, camSettings)};

  SE3 bodyToCam[RobotcarReader::numCams] = {bodyToLeft, bodyToRear,
                                            bodyToRight};
  for (int i = 0; i < RobotcarReader::numCams; ++i)
    bodyToCam[i] = RobotcarReader::camToImage * bodyToCam[i];

  return CameraBundle(bodyToCam, models, RobotcarReader::numCams);
}

void readTs(const fs::path &tsFile, std::vector<Timestamp> &timestamps) {
  CHECK(fs::exists(tsFile)) << tsFile.native() + " does not exist";
  std::ifstream ifs(tsFile);
  Timestamp ts;
  int one;
  while (ifs >> ts >> one) {
    CHECK(timestamps.empty() || timestamps.back() < ts);
    timestamps.push_back(ts);
  }
}

void readVo(const fs::path &voFile, std::vector<Timestamp> &timestamps,
            StdVector<SE3> &voBodyToFirst, bool fillVoGaps) {
  Timestamp maxSkip = -1;
  int skipCount = 0, backSkipCount = 0;
  SE3 lboToLast;

  std::ifstream ifs(voFile);
  std::string header;
  std::getline(ifs, header);
  char comma;
  Timestamp srcTs, dstTs;
  double x, y, z, roll, pitch, yaw;
  while (ifs >> srcTs >> comma >> dstTs >> comma >> x >> comma >> y >> comma >>
         z >> comma >> roll >> comma >> pitch >> comma >> yaw) {
    if (timestamps.empty()) {
      timestamps.push_back(dstTs);
      voBodyToFirst.push_back(SE3());
    }

    SE3 srcToLast = fromXyzrpy(x, y, z, roll, pitch, yaw);
    if (dstTs != timestamps.back()) {
      Timestamp skip = (dstTs - timestamps.back());
      if (skip < 0)
        backSkipCount++;
      else
        maxSkip = std::max(skip, maxSkip);
      skipCount++;

      if (fillVoGaps && timestamps.size() >= 2) {
        double tsFrac = double(skip) /
                        (timestamps.back() - timestamps[timestamps.size() - 2]);
        SE3 dstToSrcSkip = SE3::exp(tsFrac * lboToLast.log());
        srcToLast = dstToSrcSkip * srcToLast;
      }
    }
    lboToLast = srcToLast;
    SE3 srcToFirst = voBodyToFirst.back() * srcToLast;
    timestamps.push_back(srcTs);
    voBodyToFirst.push_back(srcToFirst);
  }

  LOG(INFO) << "the biggest skip in VO is " << maxSkip * 1e-6 << " seconds";
  LOG(INFO) << "there are " << skipCount << " skips in total";
}

RobotcarReader::RobotcarReader(const fs::path &_chunkDir,
                               const fs::path &modelsDir,
                               const fs::path &extrinsicsDir,
                               const ReaderSettings &_settings)
    : bodyToLeft(readBodyToCam(extrinsicsDir / "mono_left.txt"))
    , bodyToRear(readBodyToCam(extrinsicsDir / "mono_rear.txt"))
    , bodyToRight(readBodyToCam(extrinsicsDir / "mono_right.txt"))
    , bodyToLmsFront(readBodyToLidar(extrinsicsDir / "lms_front.txt"))
    , bodyToLmsRear(readBodyToLidar(extrinsicsDir / "lms_rear.txt"))
    , bodyToLdmrs(readBodyToLidar(extrinsicsDir / "ldmrs.txt"))
    , mCam(createFromData(modelsDir, bodyToLeft, bodyToRear, bodyToRight,
                          imageWidth, imageHeight, _settings.cam))
    , chunkDir(_chunkDir)
    , leftDir(chunkDir / fs::path("mono_left"))
    , rearDir(chunkDir / fs::path("mono_rear"))
    , rightDir(chunkDir / fs::path("mono_right"))
    , lmsFrontDir(chunkDir / fs::path("lms_front"))
    , lmsRearDir(chunkDir / fs::path("lms_rear"))
    , ldmrsDir(chunkDir / fs::path("ldmrs"))
    , mMasksProvided(false)
    , settings(_settings) {
  CHECK(fs::is_directory(leftDir));
  CHECK(fs::is_directory(rearDir));
  CHECK(fs::is_directory(rightDir));
  CHECK(fs::is_directory(lmsFrontDir));
  CHECK(fs::is_directory(lmsRearDir));
  CHECK(fs::is_directory(ldmrsDir));

  readTs(chunkDir / "mono_left.timestamps", mLeftTs);
  readTs(chunkDir / "mono_rear.timestamps", mRearTs);
  readTs(chunkDir / "mono_right.timestamps", mRightTs);
  readTs(chunkDir / "lms_front.timestamps", mLmsFrontTs);
  readTs(chunkDir / "lms_rear.timestamps", mLmsRearTs);
  readTs(chunkDir / "ldmrs.timestamps", mLdmrsTs);

  readVo(chunkDir / fs::path("vo") / fs::path("vo.csv"), mVoTs, voBodyToFirst,
         settings.fillVoGaps);

  syncTimestamps();
}

void RobotcarReader::provideMasks(const fs::path &masksDir) {
  CHECK(fs::is_directory(masksDir));
  fs::path leftMaskPath = masksDir / "mono_left.png";
  fs::path rearMaskPath = masksDir / "mono_rear.png";
  fs::path rightMaskPath = masksDir / "mono_right.png";
  CHECK(fs::is_regular_file(leftMaskPath));
  CHECK(fs::is_regular_file(rearMaskPath));
  CHECK(fs::is_regular_file(rightMaskPath));
  cv::Mat3b leftMask = cv::imread(std::string(leftMaskPath));
  cv::Mat3b rearMask = cv::imread(std::string(rearMaskPath));
  cv::Mat3b rightMask = cv::imread(std::string(rightMaskPath));
  cam().bundle[0].cam.setMask(cvtBgrToGray(leftMask));
  cam().bundle[1].cam.setMask(cvtBgrToGray(rearMask));
  cam().bundle[2].cam.setMask(cvtBgrToGray(rightMask));
  mMasksProvided = true;
}

int RobotcarReader::numFrames() const { return leftTs().size(); }

std::array<RobotcarReader::FrameEntry, RobotcarReader::numCams>
RobotcarReader::frame(int idx) const {
  CHECK(idx >= 0 && idx < numFrames());
  fs::path leftPath = chunkDir / fs::path("mono_left") /
                      fs::path(std::to_string(leftTs()[idx]) + ".png");
  fs::path rearPath = chunkDir / fs::path("mono_rear") /
                      fs::path(std::to_string(rearTs()[idx]) + ".png");
  fs::path rightPath = chunkDir / fs::path("mono_right") /
                       fs::path(std::to_string(rightTs()[idx]) + ".png");

  CHECK(fs::is_regular_file(leftPath));
  CHECK(fs::is_regular_file(rearPath));
  CHECK(fs::is_regular_file(rightPath));

  cv::Mat1b leftOrig = cv::imread(leftPath.native(), cv::IMREAD_GRAYSCALE);
  cv::Mat1b rearOrig = cv::imread(rearPath.native(), cv::IMREAD_GRAYSCALE);
  cv::Mat1b rightOrig = cv::imread(rightPath.native(), cv::IMREAD_GRAYSCALE);

  std::array<FrameEntry, numCams> result;
  cv::cvtColor(leftOrig, result[0].frame, cv::COLOR_BayerBG2BGR);
  result[0].timestamp = leftTs()[idx];
  cv::cvtColor(rearOrig, result[1].frame, cv::COLOR_BayerBG2BGR);
  result[1].timestamp = rearTs()[idx];
  cv::cvtColor(rightOrig, result[2].frame, cv::COLOR_BayerBG2BGR);
  result[2].timestamp = rightTs()[idx];

  return result;
}

SE3 RobotcarReader::tsToTs(Timestamp src, Timestamp dst) const {
  return tsToFirst(dst).inverse() * tsToFirst(src);
}

SE3 RobotcarReader::tsToFirst(Timestamp ts) const {
  CHECK(ts >= voTs()[0] && ts <= voTs().back())
      << "Interpolating outside of VO! "
      << "ts = " << ts << ", bounds = [" << voTs()[0] << ", " << voTs().back()
      << "]";
  CHECK(voTs().size() >= 2);
  if (ts == voTs()[0])
    return voBodyToFirst[0];

  int ind = std::lower_bound(voTs().begin(), voTs().end(), ts) - voTs().begin();
  CHECK(ind > 0 && ind < voBodyToFirst.size());
  SE3 highToLow = voBodyToFirst[ind - 1].inverse() * voBodyToFirst[ind];
  double tsFrac =
      double(ts - voTs()[ind - 1]) / (voTs()[ind] - voTs()[ind - 1]);
  SE3 tsToLow = SE3::exp(tsFrac * highToLow.log());
  return voBodyToFirst[ind - 1] * tsToLow;
}

void RobotcarReader::getPointCloudHelper(
    std::vector<Vec3> &cloud, const fs::path &scanDir, const SE3 &sensorToBody,
    Timestamp base, const std::vector<Timestamp> &timestamps, Timestamp from,
    Timestamp to, bool isLdmrs) const {
  int indFrom = std::lower_bound(timestamps.begin(), timestamps.end(), from) -
                timestamps.begin();
  int indTo = std::upper_bound(timestamps.begin(), timestamps.end(), to) -
              timestamps.begin();

  int curPerc = 0;
  int totalScans = indTo - indFrom;
  std::cout << "scanning the cloud: 0% ...";
  std::cout.flush();
  for (int i = indFrom; i < indTo; ++i) {
    SE3 sensorToBase = tsToTs(timestamps[i], base) * sensorToBody;
    fs::path scanFile =
        scanDir / fs::path(std::to_string(timestamps[i]) + ".bin");
    std::vector<double> data = readBin(scanFile);
    double x, y, z, reflectance;
    for (int i = 0; i + 2 < data.size(); i += 3) {
      if (isLdmrs) {
        x = data[i];
        y = data[i + 1];
        z = data[i + 2];
      } else {
        x = data[i];
        y = data[i + 1];
        z = 0;
        reflectance = data[i + 2];
      }
      cloud.push_back(sensorToBase * Vec3(x, y, z));
    }

    int newPerc = (i + 1 - indFrom) * 100 / totalScans;
    if (newPerc % 20 == 0 && newPerc != curPerc) {
      curPerc = newPerc;
      std::cout << " " << newPerc << "%";
      std::cout.flush();
      if (newPerc != 100)
        std::cout << " ...";
      else
        std::cout << std::endl;
    }
  }
}

std::vector<Vec3> RobotcarReader::getLmsFrontCloud(Timestamp from, Timestamp to,
                                                   Timestamp base) const {
  std::vector<Vec3> cloud;
  getPointCloudHelper(cloud, lmsFrontDir, bodyToLmsFront.inverse(), base,
                      lmsFrontTs(), from, to, false);
  return cloud;
}

std::vector<Vec3> RobotcarReader::getLmsRearCloud(Timestamp from, Timestamp to,
                                                  Timestamp base) const {
  std::vector<Vec3> cloud;
  getPointCloudHelper(cloud, lmsRearDir, bodyToLmsRear.inverse(), base,
                      lmsRearTs(), from, to, false);
  return cloud;
}

std::vector<Vec3> RobotcarReader::getLdmrsCloud(Timestamp from, Timestamp to,
                                                Timestamp base) const {
  std::vector<Vec3> cloud;
  getPointCloudHelper(cloud, ldmrsDir, bodyToLdmrs.inverse(), base, ldmrsTs(),
                      from, to, true);
  return cloud;
}

std::array<StdVector<std::pair<Vec2, double>>, RobotcarReader::numCams>
RobotcarReader::project(Timestamp from, Timestamp to, Timestamp base,
                        bool useLmsFront, bool useLmsRear, bool useLdmrs) {
  std::vector<Vec3> lmsFrontCloud;
  if (useLmsFront)
    lmsFrontCloud = getLmsFrontCloud(from, to, base);
  std::vector<Vec3> lmsRearCloud;
  if (useLmsRear)
    lmsRearCloud = getLmsRearCloud(from, to, base);
  std::vector<Vec3> ldmrsCloud;
  if (useLdmrs)
    ldmrsCloud = getLdmrsCloud(from, to, base);
  std::vector<Vec3> cloud;
  cloud.reserve(lmsFrontCloud.size() + lmsRearCloud.size() + ldmrsCloud.size());
  cloud.insert(cloud.end(), lmsFrontCloud.begin(), lmsFrontCloud.end());
  cloud.insert(cloud.end(), lmsRearCloud.begin(), lmsRearCloud.end());
  cloud.insert(cloud.end(), ldmrsCloud.begin(), ldmrsCloud.end());
  return project(cloud);
}

std::array<StdVector<std::pair<Vec2, double>>, RobotcarReader::numCams>
RobotcarReader::project(const std::vector<Vec3> &cloud) {
  std::array<StdVector<std::pair<Vec2, double>>, RobotcarReader::numCams>
      result;
  for (int camInd = 0; camInd < RobotcarReader::numCams; ++camInd) {
    SE3 bodyToCam = cam().bundle[camInd].bodyToThis;
    for (const Vec3 &p : cloud) {
      Vec3 moved = bodyToCam * p;
      if (!cam().bundle[camInd].cam.isMappable(moved))
        continue;
      double depth = moved.norm();
      Vec2 projected = cam().bundle[camInd].cam.map(moved);
      result[camInd].push_back({projected, depth});
    }
  }
  return result;
}

constexpr int npos = -1;

template <typename T>
std::vector<int> closestNotHigherInds(const std::vector<T> &a,
                                      const std::vector<T> &b) {
  CHECK(!b.empty());
  std::vector<int> inds(a.size(), npos);
  int indB = 0, indA = 0;
  while (indB < b.size() && b[indB] > a[indA])
    indA++;
  for (; indA < a.size(); ++indA) {
    if (indB + 1 == b.size() && a[indA] < b[indB])
      break;
    while (indB + 1 < b.size() && b[indB + 1] <= a[indA])
      ++indB;
    inds[indA] = indB;
  }
  return inds;
}

template <typename T>
std::vector<int> closestNotLowerInds(const std::vector<T> &a,
                                     const std::vector<T> &b) {
  CHECK(!b.empty());
  std::vector<int> inds(a.size(), npos);
  int indA = 0, indB = 0;
  while (indB < b.size() && b[indB] < a[indA])
    indB++;
  for (; indA < a.size(); ++indA) {
    while (indB < b.size() && b[indB] < a[indA])
      ++indB;
    if (indB == b.size())
      break;
    inds[indA] = indB;
  }
  return inds;
}

template <typename T>
std::vector<int> closestIndsInB(const std::vector<T> &a,
                                const std::vector<T> &b) {
  CHECK(!b.empty());
  std::vector<int> closestToB(a.size());
  int indB = 0;
  for (int indA = 0; indA < a.size(); ++indA) {
    while (indB + 1 < b.size() && b[indB + 1] < a[indA])
      ++indB;
    if (indB + 1 == b.size()) {
      closestToB[indA] = indB;
      continue;
    }
    if (std::abs(a[indA] - b[indB]) < std::abs(a[indA] - b[indB + 1]))
      closestToB[indA] = indB;
    else
      closestToB[indA] = ++indB;
  }
  return closestToB;
}

template <typename T> T triDist(T x1, T x2, T x3) {
  return abs(x1 - x2) + abs(x1 - x3) + abs(x2 - x3);
}

template <typename T>
void filterBest(std::vector<T> &a, const std::vector<T> &b,
                const std::vector<T> &c, std::vector<int> &aToB,
                std::vector<int> &aToC) {
  constexpr int npos = -1;
  std::vector<int> bestIndAInB(b.size(), npos);
  std::vector<T> bestAInB(b.size());
  for (int indA = 0; indA < a.size(); ++indA) {
    T closeB = b[aToB[indA]], closeC = c[aToC[indA]];
    T curDist = triDist(a[indA], closeB, closeC);
    if (bestIndAInB[aToB[indA]] == npos || curDist < bestAInB[aToB[indA]]) {
      bestAInB[aToB[indA]] = curDist;
      bestIndAInB[aToB[indA]] = indA;
    }
  }

  std::vector<char> removed(a.size(), false);
  for (int indA = 0; indA < a.size(); ++indA) {
    if (bestIndAInB[aToB[indA]] != indA)
      removed[indA] = true;
  }
  int newIndA = 0;
  for (int oldIndA = 0; oldIndA < a.size(); ++oldIndA)
    if (!removed[oldIndA]) {
      a[newIndA] = a[oldIndA];
      aToB[newIndA] = aToB[oldIndA];
      aToC[newIndA] = aToC[oldIndA];
      ++newIndA;
    }
  a.resize(newIndA);
  aToB.resize(newIndA);
  aToC.resize(newIndA);
}

template <typename T>
void filterOutNoRef(std::vector<T> &v, std::vector<int> &inds) {
  auto newIndsEnd = std::unique(inds.begin(), inds.end());
  CHECK_EQ(newIndsEnd - inds.begin(), inds.end() - inds.begin());
  int newInd = 0;
  for (int i : inds)
    v[newInd++] = v[i];
  v.resize(newInd);
}

template <typename T>
void checkInds(const std::vector<T> &a, const std::vector<T> &b,
               const std::vector<T> &c, int indA, int indB, int indC,
               T &bestDist, int &bestIndB, int &bestIndC) {
  if (indB == npos || indC == npos)
    return;
  T dist = triDist(a[indA], b[indB], c[indC]);
  if (dist < bestDist) {
    bestDist = dist;
    bestIndB = indB;
    bestIndC = indC;
  }
}

template <typename T>
void mostConsistentTriples(std::vector<T> &a, std::vector<T> &b,
                           std::vector<T> &c) {
  std::vector<int> lowerInB = closestNotHigherInds(a, b);
  std::vector<int> higherInB = closestNotLowerInds(a, b);
  std::vector<int> lowerInC = closestNotHigherInds(a, c);
  std::vector<int> higherInC = closestNotLowerInds(a, c);

  std::vector<int> aToB(a.size()), aToC(a.size());
  for (int indA = 0; indA < a.size(); ++indA) {
    T bestDist = std::numeric_limits<T>::max();
    int bestInB = npos, bestInC = npos;
    checkInds(a, b, c, indA, lowerInB[indA], lowerInC[indA], bestDist, bestInB,
              bestInC);
    checkInds(a, b, c, indA, lowerInB[indA], higherInC[indA], bestDist, bestInB,
              bestInC);
    checkInds(a, b, c, indA, higherInB[indA], lowerInC[indA], bestDist, bestInB,
              bestInC);
    checkInds(a, b, c, indA, higherInB[indA], higherInC[indA], bestDist,
              bestInB, bestInC);
    aToB[indA] = bestInB;
    aToC[indA] = bestInC;
  }

  filterBest(a, b, c, aToB, aToC);
  filterBest(a, c, b, aToC, aToB);
  filterOutNoRef(b, aToB);
  filterOutNoRef(c, aToC);
  CHECK_EQ(a.size(), b.size());
  CHECK_EQ(a.size(), c.size());
}

template <typename T>
double avgTriDist(const std::vector<T> &a, const std::vector<T> &b,
                  const std::vector<T> &c) {
  double sum = 0;
  for (int i = 0; i < a.size(); ++i)
    sum += triDist(a[i], b[i], c[i]);
  return sum / a.size();
}

void RobotcarReader::syncTimestamps() {
  int oldSize = mLeftTs.size();
  double oldAvgTriDist = avgTriDist(mLeftTs, mRearTs, mRightTs);
  mostConsistentTriples(mLeftTs, mRearTs, mRightTs);
  CHECK_EQ(mLeftTs.size(), mRightTs.size());
  CHECK_EQ(mLeftTs.size(), mRearTs.size());
  int newSize = mLeftTs.size();
  double newAvgTriDist = avgTriDist(mLeftTs, mRearTs, mRightTs);
  LOG(INFO) << "triples filtered out for syncing timestamps: "
            << oldSize - newSize;
  LOG(INFO) << "old avg tridist (s) = " << oldAvgTriDist / 1e6;
  LOG(INFO) << "new avg tridist (s) = " << newAvgTriDist / 1e6;
}