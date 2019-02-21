#include "util/DepthedImagePyramid.h"
#include "util/util.h"
#include <glog/logging.h>

namespace fishdso {

DepthedImagePyramid::DepthedImagePyramid(const cv::Mat1b &baseImage,
                                         const StdVector<Vec2> &points,
                                         const std::vector<double> &depthsVec,
                                         const std::vector<double> &weightsVec)
    : ImagePyramid(baseImage) {
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

  for (int il = 1; il < settingPyrLevels; ++il)
    depths[il] = pyrNUpDepth(integralWeightedDepths, integralWeights, il);
}

void DepthedImagePyramid::draw(cv::Mat3b &img) {
  const cv::Mat1d &d = depths[0];

  StdVector<Vec2> points;
  std::vector<double> dVec;
  for (int y = 0; y < d.rows; ++y)
    for (int x = 0; x < d.cols; ++x)
      if (d(y, x) > 0) {
        points.push_back(Vec2(double(x), double(y)));
        dVec.push_back(d(y, x));
      }

  // if (maxDepthCol < 2)
  setDepthColBounds(dVec);
  insertDepths(img, points, dVec, minDepthCol, maxDepthCol, false);
}

} // namespace fishdso
