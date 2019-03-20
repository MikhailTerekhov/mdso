#ifndef INCLUDE_STEREOMATCHER
#define INCLUDE_STEREOMATCHER

#include "system/CameraModel.h"
#include "system/StereoGeometryEstimator.h"
#include "util/Terrain.h"
#include "util/types.h"
#include <opencv2/opencv.hpp>
#include <vector>

namespace fishdso {

class StereoMatcher {
public:
  StereoMatcher(CameraModel *cam, const Settings::StereoMatcher &settings = {},
                const Settings::Threading &threadingSettings = {});

  SE3 match(cv::Mat frames[2], StdVector<Vec2> resPoints[2],
            std::vector<double> resDepths[2]) const;

  std::shared_ptr<Terrain> getBaseTerrain();

private:
  void createEstimations(const std::vector<cv::KeyPoint> keyPoints[2],
                         const cv::Mat decriptors[2]);

  void filterOutStillMatches(std::vector<cv::DMatch> &matches,
                             std::vector<cv::DMatch> &stillMatches,
                             const std::vector<cv::KeyPoint> kp[2]) const;

  CameraModel *cam;
  cv::Mat descriptorsMask;
  cv::Mat altMask;
  cv::Ptr<cv::ORB> orb;
  std::unique_ptr<cv::DescriptorMatcher> descriptorMatcher;

  Settings::StereoMatcher settings;
  Settings::Threading threadingSettings;
};

} // namespace fishdso

#endif
