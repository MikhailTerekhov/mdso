#include "system/keyframe.h"
#include "util/defs.h"
#include "util/settings.h"
#include "util/util.h"
#include <opencv2/opencv.hpp>

namespace fishdso {

int KeyFrame::adaptiveBlockSize = settingInitialAdaptiveBlockSize;

KeyFrame::KeyFrame(const cv::Mat &frameColored) : frameColored(frameColored), areDepthsSet(false) {
  cv::cvtColor(frameColored, framePyr[0], cv::COLOR_BGR2GRAY);
  grad(framePyr[0], gradX, gradY, gradNorm);
  selectPoints();
  setImgPyrs();
}

void KeyFrame::setDepthPyrs() {
  depths[0] = cv::Mat1f(framePyr[0].rows, framePyr[0].cols, -1.0);
  for (int i = 1; i < PL; ++i)
    depths[i] = cv::Mat1f(framePyr[i].size());
  
  cv::Mat1f weights = cv::Mat1f::zeros(framePyr[0].size());
  for (const auto &ip : interestPoints) {
    depths[0](toCvPoint(ip.p)) = ip.depth;
    weights(toCvPoint(ip.p)) = 1 / std::sqrt(ip.variance);
  }

  cv::Mat1f weightedDepths = depths[0].mul(weights, 1); 
  std::cout << "weighted depths chceck: " << weightedDepths(0, 0) << std::endl;
  cv::Mat1f integralWeightedDepths;
  cv::Mat1f integralWeights;
  cv::integral(weights, integralWeights, CV_32F);
  cv::integral(weightedDepths, integralWeightedDepths, CV_32F);
  
  for (int il = 1; il < PL; ++il) 
    depths[il] = pyrNUpDepth(integralWeightedDepths, integralWeights, il);
 
  areDepthsSet = true;
}

void KeyFrame::updateAdaptiveBlockSize(int pointsFound) {
  adaptiveBlockSize *= std::sqrt(static_cast<double>(pointsFound) /
                                 settingInterestPointsAdaptTo);
}

void selectInterestPointsInternal(const cv::Mat &gradNorm, int selBlockSize,
                                  double threshold,
                                  std::vector<cv::Point> &res) {
  for (int i = 0; i + selBlockSize < gradNorm.rows; i += selBlockSize)
    for (int j = 0; j + selBlockSize < gradNorm.cols; j += selBlockSize) {
      cv::Mat block = gradNorm(cv::Range(i, i + selBlockSize),
                               cv::Range(j, j + selBlockSize));
      double avg = cv::sum(block)[0] / (selBlockSize * selBlockSize);
      double mx = 0;
      cv::Point maxLoc = cv::Point(0, 0);
      cv::minMaxLoc(block, NULL, &mx, NULL, &maxLoc);
      if (mx > avg + threshold)
        res.push_back(cv::Point(j, i) + maxLoc);
    }
}

void KeyFrame::selectPoints() {
  std::vector<cv::Point> pointsOverThres[LI];
  std::vector<cv::Point> pointsAll;

  for (int i = 0; i < LI; ++i)
    pointsOverThres[i].reserve(settingInterestPointsAdaptTo);

  for (int i = 0; i < LI; ++i) {
    selectInterestPointsInternal(gradNorm,
                                 (1 << i) * settingInitialAdaptiveBlockSize,
                                 settingGradThreshold[i], pointsOverThres[i]);
    std::random_shuffle(pointsOverThres[i].begin(), pointsOverThres[i].end());
  }

  int foundTotal = std::accumulate(
      pointsOverThres, pointsOverThres + LI, pointsOverThres[0].size(),
      [](int accumulated, const std::vector<cv::Point> &b) {
        return accumulated + b.size();
      });

  if (foundTotal > settingInterestPointsUsed) {
    int sz = 0;
    for (int i = 1; i < LI; ++i) {
      pointsOverThres[i].resize(pointsOverThres[i].size() *
                                settingInterestPointsUsed / foundTotal);
      sz += pointsOverThres[i].size();
    }
    pointsOverThres[0].resize(settingInterestPointsUsed - sz);
  }

  for (int curL = 0; curL < LI; ++curL)
    for (cv::Point p : pointsOverThres[curL])
      interestPoints.push_back(InterestPoint(toVec2(p)));

  updateAdaptiveBlockSize(foundTotal);
}

void KeyFrame::setImgPyrs() {
  for (int lvl = 1; lvl < PL; ++lvl)
    framePyr[lvl] = boxFilterPyrUp<unsigned char>(framePyr[lvl - 1]);
}


cv::Mat KeyFrame::drawDepthedFrame(int pyrLevel, double minDepth, double maxDepth) {
  if (!areDepthsSet)
    throw std::runtime_error("trying to draw depths while they weren't initialized");
  
  int w = framePyr[pyrLevel].cols, h = framePyr[pyrLevel].rows;
  cv::Mat3b res(h, w);
  cv::cvtColor(framePyr[pyrLevel], res, cv::COLOR_GRAY2BGR);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      if (depths[pyrLevel](y, x) > 0)
        res(y, x) = toCvVec3bDummy(depthCol(depths[pyrLevel](y, x), minDepth, maxDepth));
  return res;
 }

} // namespace fishdso
