#include "system/KeyFrame.h"
#include "util/defs.h"
#include "util/settings.h"
#include "util/util.h"
#include <opencv2/opencv.hpp>

namespace fishdso {

KeyFrame::KeyFrame(CameraModel *cam, const cv::Mat &frameColored,
                   int globalFrameNum, PixelSelector &pixelSelector)
    : preKeyFrame(std::shared_ptr<PreKeyFrame>(
          new PreKeyFrame(cam, frameColored, globalFrameNum))) {
  std::vector<cv::Point> points = pixelSelector.select(
      frameColored, preKeyFrame->gradNorm, settingInterestPointsUsed, nullptr);
  addImmatures(points);
}

KeyFrame::KeyFrame(std::shared_ptr<PreKeyFrame> newPreKeyFrame,
                   PixelSelector &pixelSelector)
    : preKeyFrame(newPreKeyFrame) {
  std::vector<cv::Point> points =
      pixelSelector.select(newPreKeyFrame->frameColored, preKeyFrame->gradNorm,
                           settingInterestPointsUsed, nullptr);
  addImmatures(points);
}

void KeyFrame::addImmatures(const std::vector<cv::Point> &points) {
  immaturePoints.reserve(immaturePoints.size() + points.size());
  for (const cv::Point &p : points) {
    std::unique_ptr<ImmaturePoint> ip(
        new ImmaturePoint(preKeyFrame.get(), toVec2(p)));
    immaturePoints.insert(std::move(ip));
  }
}

void KeyFrame::selectPointsDenser(PixelSelector &pixelSelector,
                                  int pointsNeeded) {
  std::vector<cv::Point> points = pixelSelector.select(
      preKeyFrame->frameColored, preKeyFrame->gradNorm, pointsNeeded, nullptr);
  immaturePoints.clear();
  optimizedPoints.clear();
  addImmatures(points);
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
