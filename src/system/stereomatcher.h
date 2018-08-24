#pragma once

#include "system/cameramodel.h"
#include "system/stereogeometryestimator.h"
#include "util/types.h"
#include <opencv2/opencv.hpp>
#include <vector>

namespace fishdso {

class StereoMatcher {
public:
  StereoMatcher(CameraModel *cam);

  void addBaseFrame(const cv::Mat &newBaseFrame);
  void createEstimations(const cv::Mat &frame);

private:
  CameraModel *cam;

  std::unique_ptr<StereoGeometryEstimator> geometryEstimator;
  cv::Mat baseFrame;
  bool hasBaseFrame;
  cv::Ptr<cv::ORB> orb;
  std::unique_ptr<cv::DescriptorMatcher> descriptorMatcher;
  std::vector<cv::KeyPoint> baseFrameKeyPoints;
  cv::Mat baseFrameDescriptors;
  cv::Mat descriptorsMask;
};

} // namespace fishdso
