#include "util/DepthedImagePyramid.h"
#include "util/util.h"
#include <glog/logging.h>

namespace fishdso {

DepthedImagePyramid::DepthedImagePyramid(const cv::Mat1b &baseImage,
                                         int levelNum, Vec2 points[],
                                         double depthsArray[],
                                         double weightsArray[], int size)
    : ImagePyramid(baseImage, levelNum)
    , depths(levelNum) {
  depths[0] = cv::Mat1d(baseImage.rows, baseImage.cols, -1.0);
  cv::Mat1d weights = cv::Mat1d(baseImage.rows, baseImage.cols, 0.0);
  for (int i = 0; i < size; ++i) {
    cv::Point cvp = toCvPoint(points[i]);
    depths[0](cvp) = depthsArray[i];
    weights(cvp) = weightsArray[i];
  }

  cv::Mat1d weightedDepths = depths[0].mul(weights, 1);
  cv::Mat1d integralWeightedDepths;
  cv::Mat1d integralWeights;
  cv::integral(weights, integralWeights, CV_64F);
  cv::integral(weightedDepths, integralWeightedDepths, CV_64F);

  for (int il = 1; il < levelNum; ++il)
    depths[il] = pyrNUpDepth(integralWeightedDepths, integralWeights, il);
}

} // namespace fishdso
