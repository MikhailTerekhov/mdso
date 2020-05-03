#ifndef INCLUDE_FLAGS
#define INCLUDE_FLAGS

#include "util/settings.h"
#include <gflags/gflags.h>

DECLARE_int32(num_threads);

DECLARE_int32(points_per_frame);
DECLARE_int32(max_opt_points);
DECLARE_int32(max_key_frames);

DECLARE_bool(set_min_depth);
DECLARE_bool(set_max_depth);
DECLARE_bool(min_plus_exp_depth);

DECLARE_double(init_lambda);
DECLARE_double(accepted_rho);

DECLARE_bool(use_grad_weighting);

DECLARE_int32(first_frames_skip);
DECLARE_bool(run_max_RANSAC_iterations);
DECLARE_bool(average_ORB_motion);
DECLARE_bool(switch_first_motion_to_GT);

DECLARE_bool(use_alt_H_weighting);
DECLARE_int32(tracing_GN_iter);

DECLARE_double(pos_variance);

DECLARE_double(tracing_impr_factor);
DECLARE_double(epi_outlier_e);
DECLARE_double(epi_outlier_q);
DECLARE_double(optimized_stddev);

DECLARE_bool(continue_choosing_keyframes);
DECLARE_bool(track_from_lask_kf);
DECLARE_bool(predict_using_screw);
DECLARE_bool(use_grad_weights_on_tracking);
DECLARE_double(track_fail_factor);

DECLARE_bool(gt_poses);

DECLARE_bool(run_ba);
DECLARE_bool(self_written_ba);
DECLARE_string(optimization_type);
DECLARE_bool(fixed_motion_on_first_ba);
DECLARE_bool(optimize_affine_light);

DECLARE_bool(use_random_optimized_choice);

DECLARE_bool(disable_marginalization);

DECLARE_bool(deterministic);

DECLARE_int32(shift_between_keyframes);

DECLARE_bool(trivial_loss);

DECLARE_bool(dso_like);

namespace mdso {

Settings getFlaggedSettings();
}

#endif
