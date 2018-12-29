#include "system/KeyFrame.h"
#include "util/defs.h"
#include "util/settings.h"
#include "util/util.h"
#include <opencv2/opencv.hpp>

namespace fishdso {

int KeyFrame::adaptiveBlockSize = settingInitialAdaptiveBlockSize;

KeyFrame::KeyFrame(CameraModel *cam, const cv::Mat &frameColored,
                   int globalFrameNum)
    : preKeyFrame(std::shared_ptr<PreKeyFrame>(
          new PreKeyFrame(cam, frameColored, globalFrameNum))) {
  grad(preKeyFrame->frame(), gradX, gradY, gradNorm);

  int foundTotal = selectPoints(adaptiveBlockSize, settingInterestPointsUsed);
  lastBlockSize = adaptiveBlockSize;
  updateAdaptiveBlockSize(foundTotal);
}

KeyFrame::KeyFrame(std::shared_ptr<PreKeyFrame> newPreKeyFrame)
    : preKeyFrame(newPreKeyFrame) {
  grad(preKeyFrame->frame(), gradX, gradY, gradNorm);

  int foundTotal = selectPoints(adaptiveBlockSize, settingInterestPointsUsed);
  lastBlockSize = adaptiveBlockSize;
  updateAdaptiveBlockSize(foundTotal);
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

void KeyFrame::activateAllImmature() {
  for (const auto &ip : immaturePoints)
    optimizedPoints.insert(
        std::unique_ptr<OptimizedPoint>(new OptimizedPoint(*ip)));
  immaturePoints.clear();
}

void KeyFrame::deactivateAllOptimized() {
  for (const auto &op : optimizedPoints) {
    std::unique_ptr<ImmaturePoint> ip(
        new ImmaturePoint(preKeyFrame.get(), op->p));
    ip->depth = op->depth();
    immaturePoints.insert(std::move(ip));
  }
  optimizedPoints.clear();
}

std::unique_ptr<DepthedImagePyramid> KeyFrame::makePyramid() {
  std::vector<cv::Point> points =
      reservedVector<cv::Point>(optimizedPoints.size());
  std::vector<double> depths = reservedVector<double>(optimizedPoints.size());
  std::vector<double> weights = reservedVector<double>(optimizedPoints.size());

  for (const auto &op : optimizedPoints) {
    if (op->state != OptimizedPoint::ACTIVE)
      continue;
    points.push_back(toCvPoint(op->p));
    depths.push_back(op->depth());
    weights.push_back(1 / op->stddev);
  }

  return std::make_unique<DepthedImagePyramid>(preKeyFrame->frame(), points,
                                               depths, weights);
}

int KeyFrame::selectPoints(int blockSize, int pointsNeeded) {
  lastBlockSize = blockSize;
  // std::cout << "selecting with blockSize = " << blockSize << std::endl;
  std::vector<cv::Point> pointsOverThres[LI];
  std::vector<cv::Point> pointsAll;

  for (int i = 0; i < LI; ++i)
    pointsOverThres[i].reserve(settingInterestPointsAdaptTo);

  for (int i = 0; i < LI; ++i) {
    selectInterestPointsInternal(gradNorm, (1 << i) * blockSize,
                                 settingGradThreshold[i], pointsOverThres[i]);
    std::random_shuffle(pointsOverThres[i].begin(), pointsOverThres[i].end());
    // std::cout << "over thres " << i << " are " << pointsOverThres[i].size()
    // << std::endl;
  }

  int foundTotal = std::accumulate(
      pointsOverThres, pointsOverThres + LI, pointsOverThres[0].size(),
      [](int accumulated, const std::vector<cv::Point> &b) {
        return accumulated + b.size();
      });

  if (foundTotal > pointsNeeded) {
    int sz = 0;
    for (int i = 1; i < LI; ++i) {
      pointsOverThres[i].resize(pointsOverThres[i].size() * pointsNeeded /
                                foundTotal);
      sz += pointsOverThres[i].size();
    }
    pointsOverThres[0].resize(pointsNeeded - sz);
  }

  immaturePoints.clear();
  for (int curL = 0; curL < LI; ++curL)
    for (cv::Point p : pointsOverThres[curL]) {
      std::unique_ptr<ImmaturePoint> newImmature(
          new ImmaturePoint(preKeyFrame.get(), toVec2(p)));
      if (newImmature->state == ImmaturePoint::ACTIVE)
        immaturePoints.insert(std::move(newImmature));
    }

  lastPointsFound = foundTotal;
  lastPointsUsed = immaturePoints.size();
  lastBlockSize = blockSize;

  return foundTotal;
}

void KeyFrame::selectPointsDenser(int pointsNeeded) {
  int newBlockSize =
      lastBlockSize *
      std::sqrt(static_cast<double>(lastPointsFound) / pointsNeeded);
  selectPoints(newBlockSize, pointsNeeded);
  // std::cout << "after reselection = " << interestPoints.size() << std::endl;
}

cv::Mat KeyFrame::drawDepthedFrame(double minDepth, double maxDepth) {
  cv::Mat res = preKeyFrame->frameColored.clone();

  for (const auto &ip : immaturePoints)
    if (ip->state == ImmaturePoint::ACTIVE)
      putSquare(res, toCvPoint(ip->p), 5,
                toCvVec3bDummy(depthCol(ip->depth, minDepth, maxDepth)), 2);
  for (const auto &op : optimizedPoints)
    cv::circle(res, toCvPoint(op->p), 5,
               toCvVec3bDummy(depthCol(op->depth(), minDepth, maxDepth)), 2);

  return res;
}

} // namespace fishdso
