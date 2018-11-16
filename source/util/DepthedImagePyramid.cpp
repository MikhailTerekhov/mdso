#include "util/DepthedImagePyramid.h"
#include "util/util.h"
#include <glog/logging.h>

namespace fishdso {

DepthedImagePyramid::DepthedImagePyramid(const cv::Mat1b &baseImage,
                                         const std::vector<cv::Point> &points,
                                         const std::vector<double> &depthsVec,
                                         const std::vector<double> &weightsVec)
    : ImagePyramid(baseImage) {
  CHECK(points.size() == depthsVec.size() && depthsVec.size() == weightsVec.size());

  depths[0] = cv::Mat1d(baseImage.rows, baseImage.cols, -1.0);
  cv::Mat1d weights = cv::Mat1d(baseImage.rows, baseImage.cols, 0.0);
  for (int i = 0; i < points.size(); ++i) {
    depths[0](points[i]) = depthsVec[i];
    weights(points[i]) = weightsVec[i];
  }

  cv::Mat1d weightedDepths = depths[0].mul(weights, 1);
  cv::Mat1d integralWeightedDepths;
  cv::Mat1d integralWeights;
  cv::integral(weights, integralWeights, CV_64F);
  cv::integral(weightedDepths, integralWeightedDepths, CV_64F);

  for (int il = 1; il < settingPyrLevels; ++il)
    depths[il] = pyrNUpDepth(integralWeightedDepths, integralWeights, il);
}

} // namespace fishdso
