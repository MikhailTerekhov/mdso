#include "system/PreKeyFrame.h"
#include "util/util.h"
#include <opencv2/opencv.hpp>

namespace fishdso {

PreKeyFrame::PreKeyFrame(const cv::Mat &frameColored, int globalFrameNum)
    : areDepthsSet(false), globalFrameNum(globalFrameNum) {
  cv::cvtColor(frameColored, framePyr[0], cv::COLOR_BGR2GRAY);
  for (int lvl = 1; lvl < PL; ++lvl)
    framePyr[lvl] = boxFilterPyrUp<unsigned char>(framePyr[lvl - 1]);
}

void PreKeyFrame::setDepthPyrs(const cv::Mat1d &depths0,
                               const cv::Mat1d &weights) {
  depths[0] = depths0;
  for (int i = 1; i < PL; ++i)
    depths[i] = cv::Mat1d(framePyr[i].size());

  cv::Mat1d weightedDepths = depths[0].mul(weights, 1);
  cv::Mat1d integralWeightedDepths;
  cv::Mat1d integralWeights;
  cv::integral(weights, integralWeights, CV_64F);
  cv::integral(weightedDepths, integralWeightedDepths, CV_64F);

  for (int il = 1; il < PL; ++il)
    depths[il] = pyrNUpDepth(integralWeightedDepths, integralWeights, il);

  areDepthsSet = true;
}

cv::Mat PreKeyFrame::drawDepthedFrame(int pyrLevel, double minDepth,
                                      double maxDepth) {
  if (!areDepthsSet)
    throw std::runtime_error(
        "trying to draw depths while they weren't initialized");

  return fishdso::drawDepthedFrame(framePyr[pyrLevel], depths[pyrLevel],
                                   minDepth, maxDepth);
}

}; // namespace fishdso
