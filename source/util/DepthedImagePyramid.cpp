#include "util/DepthedImagePyramid.h"
#include "util/util.h"
#include <glog/logging.h>

namespace fishdso {

DepthedImagePyramid::DepthedImagePyramid(const cv::Mat1b &baseImage,
                                         int levelNum,
                                         const StdVector<Vec2> &points,
                                         const std::vector<double> &depthsVec,
                                         const std::vector<double> &weightsVec)
    : ImagePyramid(baseImage, levelNum)
    , depths(levelNum) {
  CHECK(points.size() == depthsVec.size() &&
        depthsVec.size() == weightsVec.size());

  depths[0] = cv::Mat1d(baseImage.rows, baseImage.cols, -1.0);
  cv::Mat1d weights = cv::Mat1d(baseImage.rows, baseImage.cols, 0.0);
  for (int i = 0; i < points.size(); ++i) {
    cv::Point cvp = toCvPoint(points[i]);
    depths[0](cvp) = depthsVec[i];
    weights(cvp) = weightsVec[i];
  }

  cv::Mat1d weightedDepths = depths[0].mul(weights, 1);
  cv::Mat1d integralWeightedDepths;
  cv::Mat1d integralWeights;
  cv::integral(weights, integralWeights, CV_64F);
  cv::integral(weightedDepths, integralWeightedDepths, CV_64F);

  for (int il = 1; il < levelNum; ++il)
    depths[il] = pyrNUpDepth(integralWeightedDepths, integralWeights, il);
}

cv::Mat3b DepthedImagePyramid::draw() {
  int w = images[0].cols, h = images[0].rows;
  std::vector<cv::Mat3b> depthed(images.size());
  for (int i = 0; i < images.size(); ++i) {
    int lw = images[i].cols, lh = images[i].rows;
    int s = FLAGS_rel_point_size * (lw + lh) / 2;
    cv::cvtColor(images[i], depthed[i], cv::COLOR_GRAY2BGR);
    for (int y = 0; y < images[i].rows; ++y)
      for (int x = 0; x < images[i].cols; ++x)
        if (depths[i](y, x) > -0.5)
          putSquare(depthed[i], cv::Point(x, y), s,
                    depthCol(depths[i](y, x), minDepthCol, maxDepthCol),
                    cv::FILLED);
  }
  return drawLeveled(depthed.data(), images.size(), w, h);
}

} // namespace fishdso
