#include "util/flags.h"

using namespace mdso;

DEFINE_int32(num_threads, Settings::Threading::default_numThreads,
             "Number of threads for Ceres Solver to use.");

DEFINE_int32(points_per_frame, Settings::KeyFrame::default_immaturePointsNum,
             "Number of points to trace per keyframe.");

DEFINE_int32(max_opt_points, Settings::default_maxOptimizedPoints,
             "Maximum number of points present in optimization.");

DEFINE_int32(max_key_frames, Settings::default_maxKeyFrames,
             "Maximum number of key frames to exist in the optimisation "
             "simultaneously.");

DEFINE_bool(set_min_depth, Settings::Depth::default_setMinBound,
            "Do we need to set minimum depth in ceres? This flag is shadowed "
            "if min_plus_exp_depth is enabled.");
DEFINE_bool(set_max_depth, Settings::Depth::default_setMaxBound,
            "Do we need to set maximum depth in ceres?");
DEFINE_bool(min_plus_exp_depth,
            Settings::Depth::default_useMinPlusExpParametrization,
            "If set, depths are parameterized as d = d_{min} + exp(d_p) inside "
            "of ceres, d_p being an actual parameter of optimization.");

DEFINE_double(init_lambda,
              Settings::Optimization::StepControl::default_initialLambda,
              "Initial lambda for Levenberg-Marquardt.");

DEFINE_int32(first_frames_skip,
             Settings::DelaunayDsoInitializer::default_firstFramesSkip,
             "Number of frames to skip between two frames when initializing "
             "from keypoints.");
DEFINE_bool(
    run_max_RANSAC_iterations,
    Settings::StereoMatcher::StereoGeometryEstimator::default_runMaxRansacIter,
    "Always run maximum RANSAC iterations. This will be extremely long!");
DEFINE_bool(
    average_ORB_motion,
    Settings::StereoMatcher::StereoGeometryEstimator::default_runAveraging,
    "Use NNLS motion averaging after RANSAC?");

DEFINE_bool(optimize_affine_light,
            Settings::AffineLight::default_optimizeAffineLight,
            "Perform affine light transform optimization while tracking?");

DEFINE_bool(use_alt_H_weighting,
            Settings::PointTracer::default_useAltHWeighting,
            "Do we need to use alternative formula for H robust weighting when "
            "performing subpixel tracing?");
DEFINE_int32(tracing_GN_iter, Settings::PointTracer::default_gnIter,
             "Max number of GN iterations when performing subpixel tracing. "
             "Set to 0 to disable subpixel tracing.");
DEFINE_double(pos_variance, Settings::PointTracer::default_positionVariance,
              "Expected epipolar curve placement deviation");
DEFINE_double(tracing_impr_factor, Settings::PointTracer::default_imprFactor,
              "Minimum predicted stddev improvement for tracing to happen.");
DEFINE_double(epi_outlier_e, Settings::PointTracer::default_outlierEnergyFactor,
              "Part of energy to be in outlier zone for a point to be "
              "considered outlier while tracing.");

DEFINE_bool(continue_choosing_keyframes,
            Settings::default_continueChoosingKeyFrames,
            "Do we need to use keyframes other than those we have from "
            "initialization?");
DEFINE_bool(track_from_last_kf, Settings::default_trackFromLastKf,
            "Use last keyframe as the base one for tracking? If set to false, "
            "last but one keyframe is used");
DEFINE_bool(
    predict_using_screw, Settings::default_predictUsingScrew,
    "Predict motion to the newest frame by dividing previous motion as a screw "
    "motion (use SLERP over the whole SE(3)? If set to false, SLERP is done "
    "only on rotation, and trnslational part is simply divided");
DEFINE_bool(use_grad_weights_on_tracking,
            Settings::FrameTracker::default_useGradWeighting,
            "Use gradient-dependent residual weights when tracking");
DEFINE_double(track_fail_factor,
              Settings::FrameTracker::default_trackFailFactor,
              "If RMSE after tracking another frame grew by this factor, "
              "tracking is considered failed.");

DEFINE_bool(run_ba, Settings::Optimization::default_runBA,
            "Do we need to run bundle adjustment?");

DEFINE_bool(self_written_ba,
            Settings::Optimization::default_useSelfWrittenOptimization,
            "If set to true, Ceres is not used for bundle adjustment.");

DEFINE_bool(fixed_motion_on_first_ba,
            Settings::Optimization::default_fixedMotionOnFirstAdjustent,
            "Optimize only depths when running bundle adjustment on first two "
            "keyframes? We could assume that a good motion estimation is "
            "already availible due to RANSAC initialization and averaging.");

DEFINE_double(optimized_stddev, Settings::PointTracer::default_optimizedStddev,
              "Max disparity error for a point to become optimized.");

DEFINE_bool(use_random_optimized_choice,
            Settings::default_useRandomOptimizedChoice,
            "If set to true, new OptimizedPoint-s are chosen randomly. "
            "Distance-based heuristic is used otherwise.");

DEFINE_bool(disable_marginalization, Settings::default_disableMarginalization,
            "If set to true, no keyframes are ever marginalized, despite "
            "performance issues.");

DEFINE_bool(deterministic, true,
            "Do we need deterministic random number generation?");

DEFINE_int32(shift_between_keyframes, Settings::default_keyFrameDist,
             "Difference in frame numbers between chosen keyFrames.");

DEFINE_bool(trivial_loss, false, "Use trivial loss function?");

DEFINE_bool(
    dso_like, false,
    "If set, some settings wil be overwritten to resemble original DSO setup.");

namespace mdso {

Settings getFlaggedSettings() {
  Settings settings;

  settings.threading.numThreads = FLAGS_num_threads;
  settings.keyFrame.setImmaturePointsNum(FLAGS_points_per_frame);
  settings.depth.setMinBound = FLAGS_set_min_depth;
  settings.depth.setMaxBound = FLAGS_set_max_depth;
  settings.depth.useMinPlusExpParametrization = FLAGS_min_plus_exp_depth;
  settings.optimization.stepControl.initialLambda = FLAGS_init_lambda;
  settings.delaunayDsoInitializer.firstFramesSkip = FLAGS_first_frames_skip;
  settings.stereoMatcher.stereoGeometryEstimator.runMaxRansacIter =
      FLAGS_run_max_RANSAC_iterations;
  settings.stereoMatcher.stereoGeometryEstimator.runAveraging =
      FLAGS_average_ORB_motion;
  settings.optimization.runBA = FLAGS_run_ba;
  settings.optimization.useSelfWrittenOptimization = FLAGS_self_written_ba;
  settings.affineLight.optimizeAffineLight = FLAGS_optimize_affine_light;
  settings.pointTracer.useAltHWeighting = FLAGS_use_alt_H_weighting;
  settings.pointTracer.gnIter = FLAGS_tracing_GN_iter;
  settings.pointTracer.positionVariance = FLAGS_pos_variance;
  settings.pointTracer.outlierEnergyFactor = FLAGS_epi_outlier_e;
  settings.continueChoosingKeyFrames = FLAGS_continue_choosing_keyframes;
  settings.trackFromLastKf = FLAGS_track_from_last_kf;
  settings.predictUsingScrew = FLAGS_predict_using_screw;
  settings.frameTracker.useGradWeighting = FLAGS_use_grad_weights_on_tracking;
  settings.frameTracker.trackFailFactor = FLAGS_track_fail_factor;
  settings.optimization.runBA = FLAGS_run_ba;
  settings.optimization.fixedMotionOnFirstAdjustent =
      FLAGS_fixed_motion_on_first_ba;
  settings.pointTracer.optimizedStddev = FLAGS_optimized_stddev;
  settings.useRandomOptimizedChoice = FLAGS_use_random_optimized_choice;
  settings.disableMarginalization = FLAGS_disable_marginalization;
  settings.setKeyFrameDist(FLAGS_shift_between_keyframes);
  settings.setMaxOptimizedPoints(FLAGS_max_opt_points);
  settings.setMaxKeyFrames(FLAGS_max_key_frames);
  settings.optimization.lossType = FLAGS_trivial_loss
                                       ? Settings::Optimization::TRIVIAL
                                       : Settings::Optimization::HUBER;

  if (FLAGS_dso_like)
    settings = settings.getDsoLikeSettings();

  return settings;
}

} // namespace mdso
