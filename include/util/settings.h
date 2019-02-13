#ifndef INCLUDE_SETTINGS
#define INCLUDE_SETTINGS

#include "util/types.h"
#include <gflags/gflags.h>

namespace fishdso { // candidate point selection
constexpr int settingInterestPointLayers = 3;
extern double settingGradThreshold[settingInterestPointLayers];
extern int settingInitialAdaptiveBlockSize;
extern double settingInterestPointsAdaptFactor;
extern int settingInterestPointsUsed;

// catadioptric camera projection algorithm
extern int settingCameraMapPolyDegree;
extern int settingCameraMapPolyPoints;

// orb-keypoints-based initialization
extern int settingKeyPointsCount;
extern int settingRansacMaxIter;
extern double settingInitKeypointsObserveAngle;
extern double settingMatchNonMove;
extern bool settingUsePlainTriangulation;

constexpr int settingEssentialMinimalSolveN = 5;
extern double settingEssentialSuccessProb;
extern double settingEssentialReprojErrThreshold;
extern double settingRemoveResidualsRatio;

// undistortion
extern int settingHalfFillingFilterSize;

// triangulation
extern double settingEpsPointIsOnSegment;
extern double settingEpsSamePoints;
extern double settingTriangulationDrawPadding;

// epipolar curve search
extern int settingEpipolarOnImageTestCount;
extern double settingEpipolarMaxSearchRel;
extern double settingEpipolarPositionVariance;
extern double settingEpipolarIntencityVariance;
extern double settingEpipolarMinImprovementFactor;
extern double settingEpipolarOutlierIntencityDiff;
extern double settingMinSecondBestDistance;
extern double settingOutlierEpipolarEnergy;
extern double settingOutlierEpipolarQuality;

// common direct alignment parameters
extern double settingMinAffineLigthtA;
extern double settingMaxAffineLigthtA;
extern double settingMinAffineLigthtB;
extern double settingMaxAffineLigthtB;

extern double settingMinDepth;
extern double settingMaxDepth;

extern double settingGradientWeighingConstant;

// frame tracking
constexpr int settingPyrLevels = 6;
extern double settingTrackingOutlierIntensityDiff;

// bundle adjustment
constexpr int settingResidualPatternSize = 9;
extern Vec2 settingResidualPattern[settingResidualPatternSize];
extern int settingResidualPatternHeight;
extern double settingMaxOptimizedStddev;
extern int settingMaxOptimizedPoints;
extern double settingBAOutlierIntensityDiff;
extern double settingMaxPointDepth;
extern int settingMaxFirstBAIterations;
extern int settingMaxBAIterations;

extern int settingMaxKeyFrames;

extern int settingMaxDistMapW;
extern int settingMaxDistMapH;

} // namespace fishdso

DECLARE_int32(num_threads);

DECLARE_int32(first_frames_skip);
DECLARE_bool(use_ORB_initialization);
DECLARE_bool(run_max_RANSAC_iterations);
DECLARE_bool(average_ORB_motion);
DECLARE_bool(output_reproj_CDF);
DECLARE_bool(switch_first_motion_to_GT);
DECLARE_bool(draw_inlier_matches);

DECLARE_bool(optimize_affine_light);

DECLARE_bool(perform_full_tracing);
DECLARE_bool(use_alt_H_weighting);
DECLARE_int32(tracing_GN_iter);

DECLARE_bool(perform_tracking_check_stereo);
DECLARE_bool(perform_tracking_check_GT);
DECLARE_bool(track_from_lask_kf);
DECLARE_bool(predict_using_screw);
DECLARE_bool(use_grad_weights_on_tracking);

DECLARE_bool(run_ba);
DECLARE_bool(fixed_motion_on_first_ba);

DECLARE_bool(continue_choosing_keyframes);

DECLARE_double(red_depths_part);
DECLARE_double(blue_depths_part);

DECLARE_bool(show_interpolation);
DECLARE_bool(show_track_base);
DECLARE_bool(write_files);
DECLARE_string(output_directory);

#endif
