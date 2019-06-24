#include "util/DepthedImagePyramid.h"
#include "util/util.h"
#include <glog/logging.h>

namespace fishdso {

DepthedImagePyramid::DepthedImagePyramid(const cv::Mat1b &baseImage,
                                         int levelNum,
                                         const StdVector<Point> &points)
    : ImagePyramid(baseImage, levelNum) {
  depthPyr[0].reserve(points.size());
  cv::Mat1d depths = cv::Mat1d(baseImage.rows, baseImage.cols, -1.0);
  cv::Mat1d weights = cv::Mat1d(baseImage.rows, baseImage.cols, 0.0);
  for (const Point &p : points) {
    cv::Point cvp = toCvPoint(p.p);
    if (Eigen::AlignedBox2i(Vec2i::Zero(),
                            Vec2i(baseImage.cols - 1, baseImage.rows - 1))
            .contains(Vec2i(cvp.x, cvp.y))) {
      depths(cvp) = p.depth;
      weights(cvp) = p.weight;
      depthPyr[0].push_back(p);
    }
  }

  cv::Mat1d weightedDepths = depths.mul(weights, 1);
  cv::Mat1d integralWeightedDepths;
  cv::Mat1d integralWeights;
  cv::integral(weights, integralWeights, CV_64F);
  cv::integral(weightedDepths, integralWeightedDepths, CV_64F);

  for (int il = 1; il < levelNum; ++il) {
    std::set<int> alreadyPutInd;
    int d = (1 << il);
    for (const Point &p : points) {
      cv::Point cvp = toCvPoint(p.p);
      int newX = cvp.x >> il, newY = cvp.y >> il;
      int origX = newX << il, origY = newY << il;
      if (!Eigen::AlignedBox2i(Vec2i::Zero(),
                               Vec2i(baseImage.cols - d, baseImage.rows - d))
               .contains(Vec2i(origX, origY)))
        continue;

      int ind = newX * images[il].cols + newY;
      if (alreadyPutInd.count(ind) > 0)
        continue;
      alreadyPutInd.insert(ind);

      double depthsSum = integralWeightedDepths(origY + d, origX + d) -
                         integralWeightedDepths(origY, origX + d) -
                         integralWeightedDepths(origY + d, origX) +
                         integralWeightedDepths(origY, origX);
      double weightsSum = integralWeights(origY + d, origX + d) -
                          integralWeights(origY, origX + d) -
                          integralWeights(origY + d, origX) +
                          integralWeights(origY, origX);
      if (std::abs(weightsSum) > 1e-8)
        depthPyr[il].push_back(
            {Vec2(newX, newY), depthsSum / weightsSum, weightsSum});
    }
  }
}

} // namespace fishdso
