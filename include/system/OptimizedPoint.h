#ifndef INCLUDE_INTERESTPOINT
#define INCLUDE_INTERESTPOINT

#include "system/ImmaturePoint.h"
#include "util/types.h"
#include <Eigen/Core>
#include <Eigen/StdVector>
#include <ceres/ceres.h>
#include <cmath>
#include <opencv2/core.hpp>

namespace fishdso {

struct OptimizedPoint {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  enum State { ACTIVE, OOB, OUTLIER };

  OptimizedPoint(const Vec2 &p)
      : p(p), logInvDepth(std::nan("")), variance(1), state(OUTLIER) {}
  OptimizedPoint(const ImmaturePoint &immaturePoint) : p(immaturePoint.p) {
    activate(immaturePoint.depth);
    variance = immaturePoint.variance;
  }

  EIGEN_STRONG_INLINE void activate(double depth) {
    state = ACTIVE;
    logInvDepth = -std::log(depth);
  }

  EIGEN_STRONG_INLINE double depth() const { return std::exp(-logInvDepth); }

  Vec2 p;
  double logInvDepth;
  double variance;
  State state;
};

} // namespace fishdso

#endif
