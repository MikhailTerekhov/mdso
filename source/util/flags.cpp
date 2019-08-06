#include "util/flags.h"

using namespace fishdso;

DEFINE_int32(num_threads, Settings::Threading::default_numThreads,
             "Number of threads for Ceres Solver to use.");

DEFINE_int32(points_per_frame, 2000, "Number of points to trace per keyframe.");

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

DEFINE_bool(perform_full_tracing,
            Settings::PointTracer::default_performFullTracing,
            "Do we need to search through full epipolar curve?");
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

DEFINE_bool(run_ba, Settings::BundleAdjuster::default_runBA,
            "Do we need to run bundle adjustment?");

DEFINE_bool(fixed_motion_on_first_ba,
            Settings::BundleAdjuster::default_fixedMotionOnFirstAdjustent,
            "Optimize only depths when running bundle adjustment on first two "
            "keyframes? We could assume that a good motion estimation is "
            "already availible due to RANSAC initialization and averaging.");

DEFINE_double(optimized_stddev, Settings::PointTracer::default_optimizedStddev,
              "Max disparity error for a point to become optimized.");

namespace fishdso {

Settings getFlaggedSettings() {
  Settings settings;

  settings.threading.numThreads = FLAGS_num_threads;
  settings.keyFrame.setImmaturePointsNum(FLAGS_points_per_frame);
  settings.delaunayDsoInitializer.firstFramesSkip = FLAGS_first_frames_skip;
  settings.stereoMatcher.stereoGeometryEstimator.runMaxRansacIter =
      FLAGS_run_max_RANSAC_iterations;
  settings.stereoMatcher.stereoGeometryEstimator.runAveraging =
      FLAGS_average_ORB_motion;
  settings.bundleAdjuster.runBA = FLAGS_run_ba;
  settings.affineLight.optimizeAffineLight = FLAGS_optimize_affine_light;
  settings.pointTracer.performFullTracing = FLAGS_perform_full_tracing;
  settings.pointTracer.useAltHWeighting = FLAGS_use_alt_H_weighting;
  settings.pointTracer.gnIter = FLAGS_tracing_GN_iter;
  settings.pointTracer.positionVariance = FLAGS_pos_variance;
  settings.trackFromLastKf = FLAGS_track_from_last_kf;
  settings.predictUsingScrew = FLAGS_predict_using_screw;
  settings.frameTracker.useGradWeighting = FLAGS_use_grad_weights_on_tracking;
  settings.frameTracker.trackFailFactor = FLAGS_track_fail_factor;
  settings.bundleAdjuster.runBA = FLAGS_run_ba;
  settings.bundleAdjuster.fixedMotionOnFirstAdjustent =
      FLAGS_fixed_motion_on_first_ba;
  settings.pointTracer.optimizedStddev = FLAGS_optimized_stddev;

  return settings;
}

} // namespace fishdso
