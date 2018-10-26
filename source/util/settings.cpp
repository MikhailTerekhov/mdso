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

int settingKeyPointsCount = 2000;
int settingRansacMaxIter = 100000;
double settingInitKeypointsObserveAngle = M_PI / 3;
double settingMatchNonMove = 8.0;
int settingFirstFramesSkip = 4;
double settingEssentialSuccessProb = 0.99;
double settingEssentialReprojErrThreshold = 16.0;
double settingRemoveResidualsRatio = 0.5;

int settingHalfFillingFilterSize = 1;

double settingEpsPointIsOnSegment = 1e-9;
double settingEpsSamePoints = 1e-9;

double settingTriangulationDrawPadding = 0.1;

double settingMinAffineLigthtA = -std::log(1.1);
double settingMaxAffineLigthtA = std::log(1.1);
double settingMinAffineLigthtB = -0.1 * 256;
double settingMaxAffineLigthtB = 0.1 * 256;

double settingGreadientWeighingConstant = 50.0;

double settingTrackingOutlierIntensityDiff = 12.0;
double settingBAOutlierIntensityDiff = settingTrackingOutlierIntensityDiff;
double settingMaxPointDepth = 1e4;

Vec2 settingResidualPattern[settingResidualPatternSize] = {
    Vec2(0, -2), Vec2(-1, -1), Vec2(1, -1), Vec2(-2, 0),
    Vec2(0, 0),  Vec2(2, 0),   Vec2(-1, 1), Vec2(0, 2)};

} // namespace fishdso

DEFINE_bool(use_ORB_initialization, true,
            "use keypoint-based stereomatching on first two keyframes?");
DEFINE_bool(
    output_reproj_CDF, false,
    "output reprojection errors when doing keypoint-based stereomatching? If "
    "set to true, values will be in {output_directory}/reproj_err.txt");

DEFINE_bool(optimize_affine_light, true,
            "perform affine light transform optimization while tracking?");

DEFINE_bool(perform_tracking_check, false,
            "compare tracking results with keypoint-based motion estimation?");
DEFINE_bool(track_from_lask_kf, true,
            "use last keyframe as the base one for tracking? If set to false, "
            "last but one keyframe is used");
DEFINE_bool(
    predict_using_screw, false,
    "predict motion to the newest frame by dividing previous motion as a screw "
    "motion (use SLERP over the whole SE(3)? If set to false, SLERP is done "
    "only on rotation, and trnslational part is simply divided");
DEFINE_bool(use_grad_weights_on_tracking, false,
    "use gradient-dependent residual weights when tracking");

DEFINE_bool(fixed_motion_on_first_ba, false,
            "optimize only depths when running bundle adjustment on first two "
            "keyframes? We could assume that a good motion estimation is "
            "already availible due to RANSAC initialization and averaging.");

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
