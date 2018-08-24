#include "system/stereomatcher.h"
#include "util/defs.h"
#include "util/settings.h"
#include <RelativePoseEstimator.h>

namespace fishdso {

StereoMatcher::StereoMatcher(CameraModel *cam)
    : cam(cam), hasBaseFrame(false), orb(cv::ORB::create()),
      descriptorMatcher(
          std::make_unique<cv::BFMatcher>(cv::NORM_HAMMING, true)) {
  int interestWidth, interestHeight;
  cam->getRectByAngle(settingInitKeypointsObserveAngle, interestWidth,
                      interestHeight);
  Vec2 center = cam->getImgCenter();
  double radius = cam->getImgRadiusByAngle(settingInitKeypointsObserveAngle);
  descriptorsMask = cv::Mat1b(cam->getHeight(), cam->getWidth(), CV_WHITE_BYTE);
}

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

void filterMatches(std::vector<cv::DMatch> &matches,
                   std::vector<cv::DMatch> &stillMatches,
                   const std::vector<cv::KeyPoint> &kp1,
                   const std::vector<cv::KeyPoint> &kp2) {
  stillMatches.resize(0);
  stillMatches.reserve(matches.size());

  int i = 0, j = int(matches.size()) - 1;
  while (i <= j) {
    cv::Point2f p1 = kp1[matches[i].trainIdx].pt;
    cv::Point2f p2 = kp2[matches[i].trainIdx].pt;
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

void StereoMatcher::createEstimations(const cv::Mat &frameColored) {
  cv::Mat frame = frameColored;

  std::vector<cv::KeyPoint> curFrameKeyPoints;
  cv::Mat curFrameDescriptors;
  orb->detectAndCompute(frame, cv::noArray(), curFrameKeyPoints,
                        curFrameDescriptors);
  if (curFrameKeyPoints.empty()) {
    std::cout << "no keypoints detected on the second frame!" << std::endl;
    cv::imshow("frame", frame);
  }

  std::vector<cv::DMatch> matches;
  descriptorMatcher->match(curFrameDescriptors, baseFrameDescriptors, matches);

  if (matches.empty()) {
    std::cout << "no matches found on these.." << std::endl;
    cv::imshow("img1", baseFrame);
    cv::imshow("img2", frame);
    cv::waitKey();
    return;
  }

  /*while (matches.back().distance >
         settingFeatureMatchThreshold * matches.front().distance)
    matches.pop_back();*/

  std::cout << "total matches = " << matches.size() << std::endl;

  std::vector<cv::DMatch> stillMatches;
  filterMatches(matches, stillMatches, baseFrameKeyPoints, curFrameKeyPoints);

  cv::Mat stillImg, stillImgHalfed;
  cv::drawMatches(baseFrame, baseFrameKeyPoints, frame, curFrameKeyPoints,
                  stillMatches, stillImg);
  cv::resize(stillImg, stillImgHalfed, cv::Size(), 0.5, 0.5);
  cv::imshow("still", stillImgHalfed);

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
  auto inliersMask = geometryEstimator->getInliersMask();

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

} // namespace fishdso
