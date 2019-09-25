#include "util/DepthedImagePyramid.h"
#include "util/util.h"
#include <glog/logging.h>

namespace mdso {

DepthedImagePyramid::DepthedImagePyramid(const cv::Mat1b &baseImage,
                                         int levelNum, Vec2 points[],
                                         double depthsArray[],
                                         double weightsArray[], int size)
    : ImagePyramid(baseImage, levelNum)
    , depths(levelNum) {
  std::vector<cv::Mat1d> depthsMat(levelNum);
  depthsMat[0] = cv::Mat1d(baseImage.rows, baseImage.cols, -1.0);
  cv::Mat1d weights = cv::Mat1d(baseImage.rows, baseImage.cols, 0.0);
  for (int i = 0; i < size; ++i) {
    cv::Point cvp = toCvPoint(points[i]);
    weights(cvp) = weightsArray[i];
    if (weightsArray[i] > 1e-4)
      depthsMat[0](cvp) = depthsArray[i];
  }

  cv::Mat1d weightedDepths = depthsMat[0].mul(weights, 1);
  cv::Mat1d integralWeightedDepths;
  cv::Mat1d integralWeights;
  cv::integral(weights, integralWeights, CV_64F);
  cv::integral(weightedDepths, integralWeightedDepths, CV_64F);

  for (int il = 1; il < levelNum; ++il)
    depthsMat[il] = pyrNUpDepth(integralWeightedDepths, integralWeights, il);

  for (int il = 0; il < levelNum; ++il) {
    depths[il].reserve(size);
    for (int y = 0; y < depthsMat[il].rows; ++y)
      for (int x = 0; x < depthsMat[il].cols; ++x)
        if (depthsMat[il](y, x) > 0)
          depths[il].push_back({Vec2(x, y), depthsMat[il](y, x)});
    depths.shrink_to_fit();
  }
}

} // namespace mdso
