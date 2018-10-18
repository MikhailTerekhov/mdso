#include "util/settings.h"
#include "util/defs.h"

#include <opencv2/core.hpp>

namespace fishdso {

double settingGradThreshold[settingInterestPointLayers] = {30.0, 6.0, 3.0};

int settingInitialAdaptiveBlockSize = 25;
int settingInterestPointsAdaptTo = 2150;
int settingInterestPointsUsed = 2000;

int settingCameraMapPolyDegree = 10;
int settingCameraMapPolyPoints = 2000;

int settingKeyPointsCount = 2500;
int settingRansacMaxIter = 100000;
double settingInitKeypointsObserveAngle = M_PI / 3;
double settingMatchNonMove = 7.0;
int settingFirstFramesSkip = 4;
double settingEssentialSuccessProb = 0.99;
double settingEssentialReprojErrThreshold = 4.0;
double settingRemoveResidualsRatio = 0.5;

int settingHalfFillingFilterSize = 1;

double settingEpsPointIsOnSegment = 1e-9;
double settingEpsSamePoints = 1e-9;

double settingTriangulationDrawPadding = 0.1;

double settingTrackingOutlierIntensityDiff = 12.0;
double settingBAOutlierIntensityDiff = 12.0;
double settingMaxPointDepth = 1e4;

Vec2 settingResidualPattern[settingResidualPatternSize] = {
    Vec2(0, -2), Vec2(-1, -1), Vec2(1, -1), Vec2(-2, 0),
    Vec2(0, 0),  Vec2(2, 0),   Vec2(-1, 1), Vec2(0, 2)};

} // namespace fishdso

DEFINE_bool(optimize_affine_light, true,
            "perform affine light transform optimization while tracking?");

DEFINE_bool(use_ORB_initialization, true,
            "use keypoint stereomatching on first two keyframes?");

bool validateDepthsPart(const char *flagname, double value) {
  if (value >= 0 && value <= 1)
    return true;
  std::cerr << "Invalid value for --" << std::string(flagname) << ": " << value
            << "\nit should be in [0, 1]" << std::endl;
  return false;
}

DEFINE_double(red_depths_part, 0,
              "part of contrast points that will be drawn red (i.e. they are "
              "too close to be distinguished)");
DEFINE_validator(red_depths_part, validateDepthsPart);

DEFINE_double(blue_depths_part, 0.7,
              "part of contrast points that will NOT be drawn completely blue "
              "(i.e. they are not too far to be distinguished)");
DEFINE_validator(blue_depths_part, validateDepthsPart);

DEFINE_string(output_directory, "output/default",
              "CO: \"it's output directory!\"");
