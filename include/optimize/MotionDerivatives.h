#ifndef INCLUDE_ENERGYFUNCTION
#define INCLUDE_ENERGYFUNCTION

#include "util/types.h"

namespace mdso::optimize {

struct MotionDerivatives {
  MotionDerivatives(const SE3t &hostFrameToBody, const SE3t &hostBodyToWorld,
                    const SE3t &targetBodyToWorld,
                    const SE3t &targetBodyToFrame);

  Mat34t dmatrix_dqi_host[SO3t::num_parameters];
  Mat33t d_dt_host;
  Mat34t dmatrix_dqi_target[SO3t::num_parameters];
  Mat33t d_dt_target;
};

} // namespace mdso::optimize

#endif
