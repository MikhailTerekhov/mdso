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
double settingEssentialSuccessProb = 0.999;
double settingEssentialReprojErrThreshold = 1.25;
double settingRemoveResidualsRatio = 0.5;

int settingHalfFillingFilterSize = 1;

double settingEpsPointIsOnSegment = 1e-9;
double settingEpsSamePoints = 1e-9;
double settingTriangulationDrawPadding = 0.1;

int settingEpipolarOnImageTestCount = 100;
double settingEpipolarOutlierIntencityDiff = settingBAOutlierIntensityDiff;
double settingMinSecondBestDistance = 4.0;
double settingMinOptimizedQuality = 2.0;

double settingMinAffineLigthtA = -std::log(1.1);
double settingMaxAffineLigthtA = std::log(1.1);
double settingMinAffineLigthtB = -0.1 * 256;
double settingMaxAffineLigthtB = 0.1 * 256;

double settingMinDepth = 1e-3;
double settingMaxDepth = 1e4;

double settingGradientWeighingConstant = 50.0;

double settingTrackingOutlierIntensityDiff = settingBAOutlierIntensityDiff;
double settingBAOutlierIntensityDiff = 12.0;
double settingMaxPointDepth = 1e4;
int settingMaxFirstBAIterations = 15;
int settingMaxBAIterations = 10;

int settingResidualPatternHeight = 2;
Vec2 settingResidualPattern[settingResidualPatternSize] = {
    Vec2(0, 0), Vec2(0, -2), Vec2(-1, -1), Vec2(1, -1), Vec2(-2, 0),
    Vec2(2, 0), Vec2(-1, 1), Vec2(1, 1),   Vec2(0, 2)};

int settingMaxKeyFrames = 3;

} // namespace fishdso

DEFINE_bool(use_ORB_initialization, true,
            "Use keypoint-based stereomatching on first two keyframes?");
DEFINE_bool(
    run_max_RANSAC_iterations, false,
    "Always run maximum RANSAC iterations. This will be extremely long!");
DEFINE_bool(average_ORB_motion, true,
            "Use NNLS motion averaging after RANSAC?");
DEFINE_bool(
    output_reproj_CDF, false,
    "Output reprojection errors when doing keypoint-based stereomatching? If "
    "set to true, values will be in {output_directory}/reproj_err.txt");
DEFINE_bool(switch_first_motion_to_GT, false,
            "If we have ground truth and this flag is set to true, after "
            "stereo-initialization has been performed and depths were "
            "estimated, motion we got will be replaced by the ground truth "
            "one. This is needed since stereo-estimation is poor for now, and "
            "tracking is usually performed relative to the second keyframe.");
DEFINE_bool(draw_inlier_matches, false, "Debug output stereo inlier matches.");

DEFINE_bool(optimize_affine_light, true,
            "Perform affine light transform optimization while tracking?");

DEFINE_bool(
    perform_tracking_check_stereo, false,
    "Compare tracking results with keypoint-based motion estimation? This flag "
    "will be shadowed if perform_tracking_check_GT is set to true.");
DEFINE_bool(
    perform_tracking_check_GT, false,
    "Compare tracking results with externally provided ground truth? This, if "
    "set to true, shadows perform_tracking_check_stereo. One could provide "
    "ground truth poses using DsoSystem::addGroundTruthPose. No check will "
    "happen if not all frames poses are provided.");
DEFINE_bool(track_from_lask_kf, true,
            "Use last keyframe as the base one for tracking? If set to false, "
            "last but one keyframe is used");
DEFINE_bool(
    predict_using_screw, false,
    "Predict motion to the newest frame by dividing previous motion as a screw "
    "motion (use SLERP over the whole SE(3)? If set to false, SLERP is done "
    "only on rotation, and trnslational part is simply divided");
DEFINE_bool(use_grad_weights_on_tracking, false,
            "Use gradient-dependent residual weights when tracking");

DEFINE_bool(fixed_motion_on_first_ba, false,
            "Optimize only depths when running bundle adjustment on first two "
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
              "Part of contrast points that will be drawn red (i.e. they are "
              "too close to be distinguished)");
DEFINE_validator(red_depths_part, validateDepthsPart);

DEFINE_double(blue_depths_part, 0.7,
              "Part of contrast points that will NOT be drawn completely blue "
              "(i.e. they are not too far to be distinguished)");
DEFINE_validator(blue_depths_part, validateDepthsPart);

DEFINE_string(output_directory, "output/default",
              "CO: \"it's dso's output directory!\"");
