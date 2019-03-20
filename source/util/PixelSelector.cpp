#include "util/PixelSelector.h"
#include "util/defs.h"
#include <glog/logging.h>

namespace fishdso {

PixelSelector::PixelSelector()
    : lastBlockSize(settingInitialAdaptiveBlockSize)
    , lastPointsFound(settingInterestPointsUsed) {}

std::vector<cv::Point> PixelSelector::select(const cv::Mat &frame,
                                             const cv::Mat1d &gradNorm,
                                             int pointsNeeded,
                                             cv::Mat *debugOut) {
  int newBlockSize =
      lastBlockSize *
      std::sqrt(static_cast<double>(lastPointsFound) /
                (pointsNeeded * settingInterestPointsAdaptFactor));
  return selectInternal(frame, gradNorm, pointsNeeded, newBlockSize, debugOut);
}

void selectLayer(const cv::Mat &gradNorm, int selBlockSize, double threshold,
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

std::vector<cv::Point> PixelSelector::selectInternal(const cv::Mat &frame,
                                                     const cv::Mat1d &gradNorm,
                                                     int pointsNeeded,
                                                     int blockSize,
                                                     cv::Mat *debugOut) {
  std::vector<cv::Point> pointsOverThres[LI];
  std::vector<cv::Point> pointsAll;

  for (int i = 0; i < LI; ++i)
    pointsOverThres[i].reserve(2 * pointsNeeded);

  for (int i = 0; i < LI; ++i) {
    selectLayer(gradNorm, (1 << i) * blockSize, settingGradThreshold[i],
                pointsOverThres[i]);
    std::random_shuffle(pointsOverThres[i].begin(), pointsOverThres[i].end());
    // std::cout << "over thres " << i << " are " << pointsOverThres[i].size()
    // << std::endl;
  }

  int foundTotal = std::accumulate(
      pointsOverThres, pointsOverThres + LI, pointsOverThres[0].size(),
      [](int accumulated, const std::vector<cv::Point> &b) {
        return accumulated + b.size();
      });

  std::stringstream levLog;
  for (int i = 0; i < LI - 1; ++i)
    levLog << pointsOverThres[i].size() << " + ";
  levLog << pointsOverThres[LI - 1].size();
  LOG(INFO) << "pixel selector: found " << foundTotal << " (= " << levLog.str()
            << ")" << std::endl;

  if (foundTotal > settingInterestPointsUsed) {
    int sz = 0;
    for (int i = 1; i < LI; ++i) {
      pointsOverThres[i].resize(pointsOverThres[i].size() * pointsNeeded /
                                foundTotal);
      sz += pointsOverThres[i].size();
    }
    pointsOverThres[0].resize(pointsNeeded - sz);
  }

  if (debugOut) {
    const cv::Scalar colors[LI] = {CV_GREEN, CV_BLUE, CV_RED};
    const int rad = int(5e-3 * debugOut->cols);
    for (int i = 0; i < LI; ++i)
      for (const cv::Point &p : pointsOverThres[i])
        cv::circle(*debugOut, p, rad, colors[i], 2);
  }

  for (int curL = 0; curL < LI; ++curL)
    for (const cv::Point &p : pointsOverThres[curL])
      pointsAll.push_back(p);

  lastBlockSize = blockSize;
  lastPointsFound = foundTotal;

  return pointsAll;
}

} // namespace fishdso
