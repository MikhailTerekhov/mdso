#include "util/settings.h"
#include "util/defs.h"

#include <opencv2/core.hpp>

namespace mdso {

const std::array<double, Settings::PixelSelector::gradThesholdCount>
    Settings::PixelSelector::default_gradThresholds{20.0, 8.0, 5.0};
const std::array<cv::Scalar, Settings::PixelSelector::gradThesholdCount>
    Settings::PixelSelector::default_pointColors{CV_GREEN, CV_BLUE, CV_RED};

const static_vector<Vec2, Settings::ResidualPattern::max_size>
    Settings::ResidualPattern::default_pattern{
        Vec2(0, 0), Vec2(0, -2), Vec2(-1, -1), Vec2(1, -1), Vec2(-2, 0),
        Vec2(2, 0), Vec2(-1, 1), Vec2(1, 1),   Vec2(0, 2)};

Settings Settings::getDsoLikeSettings() const {
  Settings result = *this;

  result.setMaxKeyFrames(7);
  result.setKeyFrameDist(4);
  return result;
}

Settings
Settings::getGradientAdjustedSettings(double inencityRequiredToThis,
                                      double gradNormRequiredToThis) const {
  Settings result = *this;

  for (int i = 0; i < Settings::PixelSelector::gradThesholdCount; ++i) {
    result.pixelSelector.gradThresholds[i] *= gradNormRequiredToThis;
  }
  result.residualWeighting.c *= gradNormRequiredToThis;
  result.intensity.outlierDiff *= gradNormRequiredToThis;

  return result;
}

InitializerSettings Settings::getInitializerSettings() const {
  return {delaunayDsoInitializer,
          stereoMatcher,
          threading,
          triangulation,
          keyFrame,
          {pointTracer, intensity, residualPattern, pyramid}};
}

PointTracerSettings Settings::getPointTracerSettings() const {
  return {pointTracer, intensity, residualPattern, pyramid};
}

FrameTrackerSettings Settings::getFrameTrackerSettings() const {
  return {frameTracker, pyramid,           affineLight,
          intensity,    residualWeighting, threading};
}

BundleAdjusterSettings Settings::getBundleAdjusterSettings() const {
  return {optimization,
          residualPattern,
          residualWeighting,
          intensity,
          affineLight,
          threading,
          depth,
          getEnergyFunctionSettings()};
}

ResidualSettings Settings::getResidualSettings() const {
  return {residualPattern, residualWeighting, intensity, depth};
}

EnergyFunctionSettings Settings::getEnergyFunctionSettings() const {
  return {getResidualSettings(), affineLight, optimization, depth};
}

} // namespace mdso
