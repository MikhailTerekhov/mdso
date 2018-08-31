#include "system/keyframe.h"
#include "util/defs.h"
#include "util/settings.h"
#include "util/util.h"
#include <opencv2/opencv.hpp>

namespace fishdso {

int KeyFrame::adaptiveBlockSize = settingInitialAdaptiveBlockSize;

KeyFrame::KeyFrame(const cv::Mat &frameColored) : frameColored(frameColored) {

  cv::cvtColor(frameColored, frame, cv::COLOR_BGR2GRAY);
  grad(frame, gradX, gradY, gradNorm);

  selectPoints();
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
  std::vector<cv::Point> pointsOverThres[L];
  std::vector<cv::Point> pointsAll;

  for (int i = 0; i < L; ++i)
    pointsOverThres[i].reserve(settingInterestPointsAdaptTo);

  for (int i = 0; i < L; ++i) {
    selectInterestPointsInternal(gradNorm,
                                 (1 << i) * settingInitialAdaptiveBlockSize,
                                 settingGradThreshold[i], pointsOverThres[i]);
    std::random_shuffle(pointsOverThres[i].begin(), pointsOverThres[i].end());
  }

  int foundTotal = std::accumulate(
      pointsOverThres, pointsOverThres + L, pointsOverThres[0].size(),
      [](int accumulated, const std::vector<cv::Point> &b) {
        return accumulated + b.size();
      });

  if (foundTotal > settingInterestPointsUsed) {
    int sz = 0;
    for (int i = 1; i < L; ++i) {
      pointsOverThres[i].resize(pointsOverThres[i].size() *
                                settingInterestPointsUsed / foundTotal);
      sz += pointsOverThres[i].size();
    }
    pointsOverThres[0].resize(settingInterestPointsUsed - sz);
  }

  for (int curL = 0; curL < L; ++curL)
    for (cv::Point p : pointsOverThres[curL]) {
      InterestPoint newIP;
      newIP.p = toVec2(p);
      newIP.depth = -1;
      interestPoints.push_back(newIP);
    }

  updateAdaptiveBlockSize(foundTotal);
}

} // namespace fishdso
