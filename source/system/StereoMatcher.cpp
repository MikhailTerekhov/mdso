#include "system/StereoMatcher.h"
#include "util/defs.h"
#include "util/settings.h"
#include <RelativePoseEstimator.h>
#include <glog/logging.h>

namespace fishdso {

StereoMatcher::StereoMatcher(CameraModel *cam)
    : cam(cam),
      descriptorsMask(cam->getHeight(), cam->getWidth(), CV_8U, CV_WHITE_BYTE),
      altMask(cam->getHeight(), cam->getWidth(), CV_8U, CV_BLACK_BYTE),
      orb(cv::ORB::create(settingKeyPointsCount)),
      descriptorMatcher(std::unique_ptr<cv::DescriptorMatcher>(
          new cv::BFMatcher(cv::NORM_HAMMING, true))) {

  // cv::circle(descriptorsMask, cv::Point(960, 200), 700, CV_WHITE_BYTE,
  // cv::FILLED);
  // cv::rectangle(descriptorsMask, cv::Rect(0, 200, 1920, 600), CV_WHITE_BYTE,
  // cv::FILLED);
  // cv::circle(altMask, toCvPoint(cam->getImgCenter()),
  // int(cam->getImgRadiusByAngle(M_PI_2)), CV_WHITE_BYTE, CV_FILLED);
}

void filterOutStillMatches(std::vector<cv::DMatch> &matches,
                           std::vector<cv::DMatch> &stillMatches,
                           const std::vector<cv::KeyPoint> kp[2]) {
  stillMatches.reserve(matches.size());
  stillMatches.resize(0);

  auto stillIt = std::stable_partition(
      matches.begin(), matches.end(), [kp](const cv::DMatch &m) {
        cv::Point2f p1 = kp[0][m.trainIdx].pt;
        cv::Point2f p2 = kp[1][m.queryIdx].pt;
        return cv::norm(p1 - p2) > settingMatchNonMove;
      });
  stillMatches.resize(matches.end() - stillIt);
  std::copy_n(stillIt, matches.end() - stillIt, stillMatches.begin());
  matches.erase(stillIt, matches.end());

  LOG(INFO) << "still matches removed = " << stillMatches.size() << std::endl;
}

SE3 StereoMatcher::match(cv::Mat frames[2], StdVector<Vec2> resPoints[2],
                         std::vector<double> resDepths[2]) {
  std::vector<cv::KeyPoint> keyPoints[2];
  cv::Mat descriptors[2];
  for (int i = 0; i < 2; ++i) {
    orb->detectAndCompute(frames[i], cv::noArray(), keyPoints[i],
                          descriptors[i]);
    if (keyPoints[i].empty())
      throw std::runtime_error(
          "StereoMatcher error: no keypoints found on frame " +
          std::to_string(i));
  }

  std::vector<cv::DMatch> matches;
  descriptorMatcher->match(descriptors[1], descriptors[0], matches);
  LOG(INFO) << "total matches = " << matches.size() << std::endl;
  if (matches.empty())
    throw std::runtime_error("StereoMatcher error: no matches found");

  std::vector<cv::DMatch> stillMatches;
  filterOutStillMatches(matches, stillMatches, keyPoints);

  StdVector<std::pair<Vec2, Vec2>> corresps;
  corresps.reserve(matches.size());
  for (int i = 0; i < int(matches.size()); ++i)
    corresps.push_back({toVec2(keyPoints[0][matches[i].trainIdx].pt),
                        toVec2(keyPoints[1][matches[i].queryIdx].pt)});

  std::unique_ptr<StereoGeometryEstimator> geometryEstimator =
      std::unique_ptr<StereoGeometryEstimator>(
          new StereoGeometryEstimator(cam, corresps));
  SE3 motion = geometryEstimator->findPreciseMotion();
  LOG(INFO) << "inlier matches = " << geometryEstimator->inliersNum()
            << std::endl;

  // std::vector<cv::DMatch> inlierMatches;
  // inlierMatches.reserve(matches.size());
  // for (int i : geometryEstimator->inliersInds())
    // inlierMatches.push_back(matches[i]);
  // cv::Mat imi, imi2;
  // cv::drawMatches(frames[1], keyPoints[1], frames[0], keyPoints[0],
                  // inlierMatches, imi);
  // cv::resize(imi, imi2, cv::Size(), 0.5, 0.5);
  // cv::imshow("inlier matches", imi2);

  for (int frameNum = 0; frameNum < 2; ++frameNum) {
    resPoints[frameNum].resize(0);
    resPoints[frameNum].reserve(geometryEstimator->inliersNum());
    resDepths[frameNum].resize(0);
    resDepths[frameNum].reserve(geometryEstimator->inliersNum());
  }

  for (int i : geometryEstimator->inliersInds()) {
    resPoints[0].push_back(corresps[i].first);
    resPoints[1].push_back(corresps[i].second);
    resDepths[0].push_back(geometryEstimator->depths()[i].first);
    resDepths[1].push_back(geometryEstimator->depths()[i].second);
  }

  return motion;
}

cv::Mat StereoMatcher::getMask() { return altMask; }

} // namespace fishdso
