#include "system/stereomatcher.h"
#include "util/defs.h"
#include "util/settings.h"
#include <RelativePoseEstimator.h>

namespace fishdso {

StereoMatcher::StereoMatcher(CameraModel *cam)
    : cam(cam),
      descriptorsMask(cam->getHeight(), cam->getWidth(), CV_8U, CV_BLACK_BYTE),
      orb(cv::ORB::create(2000)),
      descriptorMatcher(std::unique_ptr<cv::DescriptorMatcher>(
          new cv::BFMatcher(cv::NORM_HAMMING, true))) {

  cv::circle(descriptorsMask, cv::Point(960, 200), 700, CV_WHITE_BYTE,
             cv::FILLED);
  cv::rectangle(descriptorsMask, cv::Rect(0, 200, 1920, 600), CV_WHITE_BYTE,
                cv::FILLED);
  // cv::imshow("mask", descriptorsMask);
}

void filterOutStillMatches(std::vector<cv::DMatch> &matches,
                           std::vector<cv::DMatch> &stillMatches,
                           const std::vector<cv::KeyPoint> kp[2]) {
  stillMatches.resize(0);
  stillMatches.reserve(matches.size());

  int i = 0, j = int(matches.size()) - 1;
  while (i <= j) {
    cv::Point2f p1 = kp[0][matches[i].trainIdx].pt;
    cv::Point2f p2 = kp[1][matches[i].trainIdx].pt;
    cv::Point2f diff = p1 - p2;
    float dist = diff.x * diff.x + diff.y * diff.y;
    if (dist < settingMatchNonMove) {
      std::swap(matches[i], matches[j]);
      stillMatches.push_back(matches[j]);
      --j;
    } else {
      ++i;
    }
  }

  std::cout << "still matches removed = " << matches.size() - j - 1
            << std::endl;

  matches.resize(j + 1);
}

SE3 StereoMatcher::match(cv::Mat frames[2], std::vector<Vec2> resPoints[2],
                         std::vector<double> resDepths[2]) {
  cv::Ptr<cv::ORB> orb = cv::ORB::create(2000);
  std::vector<cv::KeyPoint> keyPoints[2];
  cv::Mat descriptors[2];
  for (int i = 0; i < 2; ++i) {
    orb->detectAndCompute(frames[i], descriptorsMask, keyPoints[i],
                          descriptors[i]);
    if (keyPoints[i].empty())
      throw std::runtime_error(
          "StereoMatcher error: no keypoints found on frame " +
          std::to_string(i));
  }

  std::vector<cv::DMatch> matches;
  descriptorMatcher->match(descriptors[1], descriptors[0], matches);
  std::cout << "total matches = " << matches.size() << std::endl;
  if (matches.empty())
    throw std::runtime_error("StereoMatcher error: no matches found");

  std::vector<cv::DMatch> stillMatches;
  filterOutStillMatches(matches, stillMatches, keyPoints);

  std::vector<std::pair<Vec2, Vec2>> corresps;
  corresps.reserve(matches.size());
  for (int i = 0; i < int(matches.size()); ++i)
    corresps.push_back({toVec2(keyPoints[0][matches[i].trainIdx].pt),
                        toVec2(keyPoints[1][matches[i].queryIdx].pt)});

  std::unique_ptr<StereoGeometryEstimator> geometryEstimator =
      std::unique_ptr<StereoGeometryEstimator>(
          new StereoGeometryEstimator(cam, corresps));
  SE3 motion = geometryEstimator->findPreciseMotion();
  std::cout << "inlier matches = " << geometryEstimator->getInliersNum()
            << std::endl;
  for (int frameNum = 0; frameNum < 2; ++frameNum) {
    resPoints[frameNum].resize(0);
    resPoints[frameNum].reserve(geometryEstimator->getInliersNum());
    resDepths[frameNum].resize(0);
    resDepths[frameNum].reserve(geometryEstimator->getInliersNum());
  }

  for (int i = 0; i < int(matches.size()); ++i) {
    if (!geometryEstimator->getInliersMask()[i])
      continue;
    resPoints[0].push_back(corresps[i].first);
    resPoints[1].push_back(corresps[i].second);
    resDepths[0].push_back(geometryEstimator->depths()[i].first);
    resDepths[1].push_back(geometryEstimator->depths()[i].second);
  }

  return motion;
}

/*
void fishdso::StereoMatcher::addBaseFrame(const cv::Mat &newBaseFrame) {
  baseFrame = newBaseFrame;

  orb->detectAndCompute(baseFrame, descriptorsMask, baseFrameKeyPoints,
                        baseFrameDescriptors);
  if (baseFrameKeyPoints.empty()) {
    cv::imshow("base frame", baseFrame);
    cv::waitKey();
    throw std::runtime_error("no keypoints detected on the base frame!");
  }
  hasBaseFrame = true;
}

std::shared_ptr<Terrain> StereoMatcher::getBaseTerrain() { return refTerrain; }

void StereoMatcher::createEstimations(const cv::Mat &frameColored) {
  cv::Mat frame = frameColored;

  std::vector<cv::KeyPoint> curFrameKeyPoints;
  cv::Mat curFrameDescriptors;
  orb->detectAndCompute(frame, descriptorsMask, curFrameKeyPoints,
                        curFrameDescriptors);
  if (curFrameKeyPoints.empty()) {
    std::cout << "no keypoints detected on the second frame!" << std::endl;
    cv::imshow("frame", frame);
  }

  std::vector<cv::DMatch> matches;
  descriptorMatcher->match(curFrameDescriptors, baseFrameDescriptors, matches);
  std::vector<cv::KeyPoint> kp1, kp2;
  for (auto mt : matches) {
    kp1.push_back(baseFrameKeyPoints[mt.trainIdx]);
    kp2.push_back(curFrameKeyPoints[mt.queryIdx]);
  }

  cv::Mat img1 = baseFrame.clone(), img2 = frame.clone();
  for (auto p : kp1) {
    cv::circle(img1, p.pt, 3, CV_RED);
  }
  for (auto p : kp2) {
    cv::circle(img2, p.pt, 3, CV_RED);
  }
  cv::Mat img12, img22;
  cv::resize(img1, img12, cv::Size(), 0.5, 0.5);
  cv::resize(img2, img22, cv::Size(), 0.5, 0.5);

  cv::imshow("kp1", img12);
  cv::imshow("kp2", img22);

  if (matches.empty()) {
    std::cout << "no matches found on these.." << std::endl;
    cv::imshow("img1", baseFrame);
    cv::imshow("img2", frame);
    cv::waitKey();
    return;
  }
  std::cout << "total matches = " << matches.size() << std::endl;

  std::vector<cv::DMatch> stillMatches;
  filterMatches(matches, stillMatches, baseFrameKeyPoints, curFrameKeyPoints);

  //  cv::Mat stillImg, stillImgHalfed;
  //  cv::drawMatches(baseFrame, baseFrameKeyPoints, frame, curFrameKeyPoints,
  //                  stillMatches, stillImg);
  //  cv::resize(stillImg, stillImgHalfed, cv::Size(), 0.5, 0.5);
  //  cv::imshow("still", stillImgHalfed);

  std::vector<std::pair<Vec2, Vec2>> imgCorresps;
  imgCorresps.reserve(matches.size());
  Vec2 baseFramePtVec, curFramePtVec;
  for (int i = 0; i < int(matches.size()); ++i) {
    cv::Point2f baseFramePt = baseFrameKeyPoints[matches[i].trainIdx].pt;
    cv::Point2f curFramePt = curFrameKeyPoints[matches[i].queryIdx].pt;

    baseFramePtVec[0] = double(baseFramePt.x);
    baseFramePtVec[1] = double(baseFramePt.y);
    curFramePtVec[0] = double(curFramePt.x);
    curFramePtVec[1] = double(curFramePt.y);

    imgCorresps.push_back({baseFramePtVec, curFramePtVec});
  }

  geometryEstimator =
      std::make_unique<StereoGeometryEstimator>(cam, imgCorresps);
  SE3 motion = geometryEstimator->findPreciseMotion();
  // SE3 motion;
  auto inliersMask = geometryEstimator->getInliersMask();
  auto depths = geometryEstimator->depths();

  //  for (char &a : inliersMask)
  //    a = 1;
  //  for (auto &d : depths)
  //    d.first = d.second = 1;

  //  std::cout << "R =\n"
  //            << motion.rotationMatrix()
  //            << "\nt = " << motion.translation().transpose()
  //            << "\nquat = " << motion.unit_quaternion().w() << " + ("
  //            << motion.unit_quaternion().vec().transpose() << ")\n";

  //  std::cout << "rotation angle = "
  //            << 2 * std::acos((motion.unit_quaternion().w()));

  std::vector<cv::DMatch> myGoodMatches;
  myGoodMatches.reserve(matches.size());
  for (int i = 0; i < int(matches.size()); ++i)
    if (inliersMask[i])
      myGoodMatches.push_back(matches[i]);

  std::vector<Vec2> newFrameKeyPoints;
  std::vector<double> newFrameDepths;
  newFrameKeyPoints.reserve(myGoodMatches.size());

  for (int i = 0; i < int(matches.size()); ++i)
    if (inliersMask[i]) {
      newFrameKeyPoints.push_back(imgCorresps[i].second);
      newFrameDepths.push_back(depths[i].second);
      // std::cout << "push d = " << depths[i].second << std::endl;
    }

  int padding = int(double(newFrameDepths.size()) * 0.2);

  std::vector<std::pair<Vec2, double>> pd(newFrameKeyPoints.size());

  for (int i = 0; i < int(newFrameKeyPoints.size()); ++i)
    pd[i] = {newFrameKeyPoints[i], newFrameDepths[i]};

  std::sort(pd.begin(), pd.end(),
            [](auto a, auto b) { return a.second < b.second; });

  double minDepth = pd[0].second;
  double maxDepth = pd[padding].second;

  refTerrain = std::shared_ptr<Terrain>(
      new Terrain(cam, newFrameKeyPoints, newFrameDepths));

  cv::Mat terrainImg = refTerrain->draw(frame);
  //  cv::Mat terrainImg2;
  //  cv::resize(cumulativeImg, terrainImg2, cv::Size(), 0.5, 0.5);
  //  cv::imshow("triang", terrainImg2);

  std::mt19937 mt;
  std::uniform_real_distribution<double> distrx(0, frame.cols);
  std::uniform_real_distribution<double> distry(0, frame.rows);
  std::vector<Vec2> randPnt;
  randPnt.reserve(settingInterestPointsUsed);
  std::vector<double> randPntDepths;
  randPntDepths.reserve(settingInterestPointsUsed);
  Vec3 resVec;

  for (int i = 0; i < settingInterestPointsUsed; ++i) {

    Vec2 newPnt(distrx(mt), distry(mt));
    if (refTerrain->operator()(newPnt, resVec)) {
      randPnt.push_back(newPnt);
      // std::cout << "put d = " << resVec.norm() << std::endl;
      randPntDepths.push_back(resVec.norm());
    }
  }

  cv::Mat kpDepths = insertDepths(terrainImg, newFrameKeyPoints, newFrameDepths,
                                  minDepth, maxDepth, true);
  //  cv::Mat dHalved;
  //  cv::resize(newframeWithDepths, dHalved, cv::Size(), 0.5, 0.5);
  //  cv::imshow("depths", dHalved);

  cv::Mat interpolatedDepths =
      insertDepths(kpDepths, randPnt, randPntDepths, minDepth, maxDepth, false);
  cv::Mat intD2;
  cv::resize(interpolatedDepths, intD2, cv::Size(), 0.5, 0.5);
  // cv::circle(interpolatedDepths, cv::Point(1650, 700), 7, CV_BLACK,
  // cv::FILLED);

  cv::imshow("interpolated", intD2);

  std::cout << "inlier matches = " << myGoodMatches.size() << std::endl;

  std::cout << "translation found = " << motion.translation().transpose()
            << "\nquaternion = "
            << motion.unit_quaternion().coeffs().transpose() << std::endl;

  cv::Mat myMatchesDrawn;
  cv::drawMatches(frame, curFrameKeyPoints, baseFrame, baseFrameKeyPoints,
                  myGoodMatches, myMatchesDrawn);
  cv::Mat myMatchesResized;
  cv::resize(myMatchesDrawn, myMatchesResized, cv::Size(), 0.5, 0.5);
  cv::imshow("inlier matches", myMatchesResized);

  cv::waitKey();
}
*/

} // namespace fishdso
