#ifndef INCLUDE_INTERESTPOINT
#define INCLUDE_INTERESTPOINT

#include "util/types.h"
#include <Eigen/Core>
#include <Eigen/StdVector>
#include <ceres/ceres.h>
#include <cmath>
#include <opencv2/core.hpp>

namespace fishdso {

struct InterestPoint {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  enum State { ACTIVE, OOB, OUTLIER };

  InterestPoint(const Vec2 &p) : p(p), state(OUTLIER) {}

  EIGEN_STRONG_INLINE void activate(double depth) {
    state = ACTIVE;
    logInvDepth = -std::log(depth);
  }

  EIGEN_STRONG_INLINE double depthd() const { return std::exp(-logInvDepth); }

  Vec2 p;
  double logInvDepth;
  double variance;
  State state;
};

} // namespace fishdso

#endif
