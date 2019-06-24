#include "system/KeyFrame.h"
#include "util/defs.h"
#include "util/settings.h"
#include "util/util.h"
#include <opencv2/opencv.hpp>

namespace fishdso {

KeyFrame::KeyFrame(CameraModel *cam, const cv::Mat &frameColored,
                   int globalFrameNum, PixelSelector &pixelSelector,
                   const Settings::KeyFrame &_kfSettings,
                   const PointTracerSettings tracingSettings)
    : preKeyFrame(std::shared_ptr<PreKeyFrame>(
          new PreKeyFrame(cam, frameColored, globalFrameNum)))
    , immaturePoints(
          reservedVector<std::unique_ptr<ImmaturePoint>>(_kfSettings.pointsNum))
    , optimizedPoints(reservedVector<std::unique_ptr<OptimizedPoint>>(
          _kfSettings.pointsNum))
    , kfSettings(_kfSettings)
    , tracingSettings(tracingSettings) {
  std::vector<cv::Point> points = pixelSelector.select(
      frameColored, preKeyFrame->gradNorm, kfSettings.pointsNum, nullptr);
  addImmatures(points);
}

KeyFrame::KeyFrame(std::shared_ptr<PreKeyFrame> newPreKeyFrame,
                   PixelSelector &pixelSelector,
                   const Settings::KeyFrame &_kfSettings,
                   const PointTracerSettings &tracingSettings)
    : preKeyFrame(newPreKeyFrame)
    , immaturePoints(
          reservedVector<std::unique_ptr<ImmaturePoint>>(_kfSettings.pointsNum))
    , optimizedPoints(reservedVector<std::unique_ptr<OptimizedPoint>>(
          _kfSettings.pointsNum))
    , kfSettings(_kfSettings)
    , tracingSettings(tracingSettings) {
  std::vector<cv::Point> points =
      pixelSelector.select(newPreKeyFrame->frameColored, preKeyFrame->gradNorm,
                           kfSettings.pointsNum, nullptr);
  addImmatures(points);
}

void KeyFrame::addImmatures(const std::vector<cv::Point> &points) {
  immaturePoints.reserve(immaturePoints.size() + points.size());
  for (const cv::Point &p : points)
    immaturePoints.push_back(std::unique_ptr<ImmaturePoint>(
        new ImmaturePoint(preKeyFrame.get(), toVec2(p), tracingSettings)));
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
    optimizedPoints.push_back(
        std::unique_ptr<OptimizedPoint>(new OptimizedPoint(*ip)));
  immaturePoints.clear();
}

void KeyFrame::deactivateAllOptimized() {
  for (const auto &op : optimizedPoints) {
    std::unique_ptr<ImmaturePoint> ip(
        new ImmaturePoint(preKeyFrame.get(), op->p, tracingSettings));
    ip->depth = op->depth();
    immaturePoints.push_back(std::move(ip));
  }
  optimizedPoints.clear();
}

cv::Mat3b KeyFrame::drawDepthedFrame(double minDepth, double maxDepth) const {
  cv::Mat res = preKeyFrame->frameColored.clone();

  for (const auto &ip : immaturePoints)
    if (ip->state == ImmaturePoint::ACTIVE && ip->maxDepth != INF)
      putSquare(res, toCvPoint(ip->p), 5,
                toCvVec3bDummy(depthCol(ip->depth, minDepth, maxDepth)), 2);
  for (const auto &op : optimizedPoints)
    cv::circle(res, toCvPoint(op->p), 5,
               toCvVec3bDummy(depthCol(op->depth(), minDepth, maxDepth)), 2);

  return res;
}

} // namespace fishdso
