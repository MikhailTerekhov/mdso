#pragma once

#include "system/cameramodel.h"
#include "system/stereogeometryestimator.h"
#include "system/terrain.h"
#include "util/types.h"
#include <opencv2/opencv.hpp>
#include <vector>

namespace fishdso {

class StereoMatcher {
public:
  StereoMatcher(CameraModel *cam);

  SE3 match(cv::Mat frames[2], stdvectorVec2 resPoints[2],
            std::vector<double> resDepths[2]);

  std::shared_ptr<Terrain> getBaseTerrain();

  cv::Mat getMask();

private:
  void createEstimations(const std::vector<cv::KeyPoint> keyPoints[2],
                         const cv::Mat decriptors[2]);
  CameraModel *cam;
  cv::Mat descriptorsMask;
  cv::Mat altMask;
  cv::Ptr<cv::ORB> orb;
  std::unique_ptr<cv::DescriptorMatcher> descriptorMatcher;
};

} // namespace fishdso
