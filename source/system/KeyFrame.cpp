#include "system/KeyFrame.h"
#include "util/defs.h"
#include "util/settings.h"
#include "util/util.h"
#include <opencv2/opencv.hpp>

namespace fishdso {

KeyFrame::KeyFrame(CameraModel *cam, const cv::Mat &frameColored,
                   int globalFrameNum, PixelSelector &pixelSelector,
                   const Settings::KeyFrame &kfSettings,
                   const PointTracerSettings tracingSettings)
    : preKeyFrame(std::shared_ptr<PreKeyFrame>(
          new PreKeyFrame(cam, frameColored, globalFrameNum)))
    , kfSettings(kfSettings)
    , tracingSettings(tracingSettings) {
  std::vector<cv::Point> points = pixelSelector.select(
      frameColored, preKeyFrame->gradNorm, kfSettings.pointsNum, nullptr);
  addImmatures(points);
}

KeyFrame::KeyFrame(std::shared_ptr<PreKeyFrame> newPreKeyFrame,
                   PixelSelector &pixelSelector,
                   const Settings::KeyFrame &kfSettings,
                   const PointTracerSettings &tracingSettings)
    : preKeyFrame(newPreKeyFrame)
    , kfSettings(kfSettings)
    , tracingSettings(tracingSettings) {
  std::vector<cv::Point> points =
      pixelSelector.select(newPreKeyFrame->frameColored, preKeyFrame->gradNorm,
                           kfSettings.pointsNum, nullptr);
  addImmatures(points);
}

void KeyFrame::addImmatures(const std::vector<cv::Point> &points) {
  immaturePoints.reserve(immaturePoints.size() + points.size());
  for (const cv::Point &p : points) {
    SetUniquePtr<ImmaturePoint> ip(
        new ImmaturePoint(preKeyFrame.get(), toVec2(p), tracingSettings));
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
        SetUniquePtr<OptimizedPoint>(new OptimizedPoint(*ip)));
  immaturePoints.clear();
}

void KeyFrame::deactivateAllOptimized() {
  for (const auto &op : optimizedPoints) {
    SetUniquePtr<ImmaturePoint> ip(
        new ImmaturePoint(preKeyFrame.get(), op->p, tracingSettings));
    ip->depth = op->depth();
    immaturePoints.insert(std::move(ip));
  }
  optimizedPoints.clear();
}

std::unique_ptr<DepthedImagePyramid> KeyFrame::makePyramid() {
  StdVector<Vec2> points;
  points.reserve(optimizedPoints.size());
  std::vector<double> depths;
  depths.reserve(optimizedPoints.size());
  std::vector<double> weights;
  weights.reserve(optimizedPoints.size());

  for (const auto &op : optimizedPoints) {
    if (op->state != OptimizedPoint::ACTIVE)
      continue;
    points.push_back(op->p);
    depths.push_back(op->depth());
    weights.push_back(1 / op->stddev);
  }

  return std::make_unique<DepthedImagePyramid>(preKeyFrame->frame(),
                                               tracingSettings.pyramid.levelNum,
                                               points, depths, weights);
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
