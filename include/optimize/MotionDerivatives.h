#ifndef INCLUDE_MOTIONDERIVATIVES
#define INCLUDE_MOTIONDERIVATIVES

#include "util/types.h"

namespace mdso::optimize {

struct MotionDerivatives {
  MotionDerivatives(const SE3t &hostFrameToBody, const SE3t &hostBodyToWorld,
                    const SE3t &targetBodyToWorld,
                    const SE3t &targetBodyToFrame);

  inline Mat34t daction_dq_host(const Vec4t &vH) const {
    return daction_dq(dmatrix_dqi_host, vH);
  }

  inline Mat34t daction_dq_target(const Vec4t &vH) const {
    return daction_dq(dmatrix_dqi_target, vH);
  }

  Mat34t dmatrix_dqi_host[SO3t::num_parameters];
  Mat33t daction_dt_host;
  Mat34t dmatrix_dqi_target[SO3t::num_parameters];
  Mat33t daction_dt_target;

private:
  static Mat34t daction_dq(const Mat34t *dmatrix_dq, const Vec4t &vH);
};

} // namespace mdso::optimize

#endif
