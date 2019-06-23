#include "util/settings.h"
#include "util/defs.h"

#include <opencv2/core.hpp>

namespace fishdso {

const std::vector<double> Settings::PixelSelector::default_gradThresholds{
    20.0, 8.0, 5.0};
const std::vector<cv::Scalar> Settings::PixelSelector::default_pointColors{
    CV_GREEN, CV_BLUE, CV_RED};
const StdVector<Vec2> Settings::ResidualPattern::default_pattern{
    Vec2(0, 0), Vec2(0, -2), Vec2(-1, -1), Vec2(1, -1), Vec2(-2, 0),
    Vec2(2, 0), Vec2(-1, 1), Vec2(1, 1),   Vec2(0, 2)};

InitializerSettings Settings::getInitializerSettings() const {
  return {delaunayDsoInitializer,
          stereoMatcher,
          threading,
          triangulation,
          keyFrame,
          {pointTracer, intencity, residualPattern, pyramid}};
}

PointTracerSettings Settings::getPointTracerSettings() const {
  return {pointTracer, intencity, residualPattern, pyramid};
}

FrameTrackerSettings Settings::getFrameTrackerSettings() const {
  return {frameTracker, pyramid,       affineLight,
          intencity,    gradWeighting, threading};
}

BundleAdjusterSettings Settings::getBundleAdjusterSettings() const {
  return {bundleAdjuster, residualPattern, gradWeighting, intencity,
          affineLight,    threading,       depth};
}

} // namespace fishdso
