#pragma once

#include "../util/types.h"
#include "cameramodel.h"
#include <opencv2/opencv.hpp>
#include <vector>

namespace fishdso {

class StereoMatcher {
public:
  StereoMatcher(CameraModel *cam);

  void addBaseFrame(const cv::Mat &newBaseFrame);
  void createEstimations(const cv::Mat &frame);

private:
  EIGEN_STRONG_INLINE int
  findInliers(const Mat33 &E, const std::vector<std::pair<Vec3, Vec3>> &rays,
              const std::vector<std::pair<Vec2, Vec2>> &projectedRays,
              std::vector<char> &inliersMask);
  EIGEN_STRONG_INLINE SE3
  extractMotion(const Mat33 &E, const std::vector<std::pair<Vec3, Vec3>> &rays,
                std::vector<char> &inliersMask, int &newInliers);

  SE3 estimateMotion(std::vector<std::pair<Vec3, Vec3>> &rays,
                     const std::vector<std::pair<Vec2, Vec2>> &projectedRays,
                     std::vector<char> &inliersMask);

  CameraModel *cam;

  cv::Rect rectOfInterest;
  cv::Mat baseFrame;
  bool hasBaseFrame;
  cv::Ptr<cv::ORB> orb;
  cv::Ptr<cv::DescriptorMatcher> descriptorMatcher;
  std::vector<cv::KeyPoint> baseFrameKeyPoints;
  cv::Mat baseFrameDescriptors;
  cv::Mat descriptorsMask;
};

} // namespace fishdso
