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
  std::cout << "inlier matches = " << geometryEstimator->inliersNum()
            << std::endl;
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

} // namespace fishdso
