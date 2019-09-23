#include "util/PixelSelector.h"
#include "util/defs.h"
#include "util/flags.h"
#include <glog/logging.h>
#include <random>

namespace mdso {

#define LI (Settings::PixelSelector::gradThesholdCount)

PixelSelector::PixelSelector(const Settings::PixelSelector &_settings)
    : lastBlockSize(_settings.initialAdaptiveBlockSize)
    , lastPointsFound(_settings.initialPointsFound)
    , settings(_settings) {}

PixelSelector::PointVector PixelSelector::select(const cv::Mat &frame,
                                                 const cv::Mat1d &gradNorm,
                                                 int pointsNeeded,
                                                 cv::Mat *debugOut) {
  int newBlockSize =
      lastBlockSize * std::sqrt(static_cast<double>(lastPointsFound) /
                                (pointsNeeded * settings.adaptToFactor));
  return selectInternal(frame, gradNorm, pointsNeeded, newBlockSize, debugOut);
}

void selectLayer(const cv::Mat &gradNorm, int selBlockSize, double threshold,
                 PixelSelector::PointVector &res) {
  for (int i = 0;
       i + selBlockSize < gradNorm.rows && res.size() < res.capacity();
       i += selBlockSize)
    for (int j = 0;
         j + selBlockSize < gradNorm.cols && res.size() < res.capacity();
         j += selBlockSize) {
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

PixelSelector::PointVector
PixelSelector::selectInternal(const cv::Mat &frame, const cv::Mat1d &gradNorm,
                              int pointsNeeded, int blockSize,
                              cv::Mat *debugOut) {
  PointVector pointsOverThres[LI];
  PointVector pointsAll;

  for (int i = 0; i < LI; ++i) {
    selectLayer(gradNorm, (1 << i) * blockSize, settings.gradThresholds[i],
                pointsOverThres[i]);

    std::mt19937 mt(FLAGS_deterministic ? 42 : std::random_device()());
    std::shuffle(pointsOverThres[i].begin(), pointsOverThres[i].end(), mt);
    // std::cout << "over thres " << i << " are " << pointsOverThres[i].size()
    // << std::endl;
  }

  int foundTotal = std::accumulate(pointsOverThres, pointsOverThres + LI, 0,
                                   [](int accumulated, const PointVector &b) {
                                     return accumulated + b.size();
                                   });

  std::stringstream levLog;
  for (int i = 0; i < LI - 1; ++i)
    levLog << pointsOverThres[i].size() << " + ";
  levLog << pointsOverThres[LI - 1].size();
  LOG(INFO) << "selector: found " << foundTotal << " (= " << levLog.str() << ")"
            << std::endl;

  if (foundTotal > pointsNeeded) {
    int sz = 0;
    for (int i = 1; i < LI; ++i) {
      pointsOverThres[i].resize(pointsOverThres[i].size() * pointsNeeded /
                                foundTotal);
      sz += pointsOverThres[i].size();
    }
    pointsOverThres[0].resize(pointsNeeded - sz);
  }

  if (debugOut) {
    cv::Mat &result = *debugOut;
    result = frame.clone();
    const int rad = int(settings.relDebugPointRadius * debugOut->cols);
    for (int i = 0; i < std::min(int(LI), 3); ++i)
      for (const cv::Point &p : pointsOverThres[i])
        cv::circle(*debugOut, p, rad, settings.pointColors[i], 2);
  }

  for (int curL = 0; curL < LI; ++curL)
    for (const cv::Point &p : pointsOverThres[curL])
      pointsAll.push_back(p);

  lastBlockSize = blockSize;
  lastPointsFound = foundTotal;

  return pointsAll;
}

} // namespace mdso
