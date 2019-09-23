#ifndef INCLUDE_INTERESTPOINT
#define INCLUDE_INTERESTPOINT

#include "system/ImmaturePoint.h"
#include "util/types.h"
#include <Eigen/Core>
#include <Eigen/StdVector>
#include <cmath>
#include <opencv2/core.hpp>

namespace mdso {

struct OptimizedPoint {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  enum State { ACTIVE, OOB, OUTLIER };

  OptimizedPoint(const ImmaturePoint &immaturePoint)
      : p(immaturePoint.p)
      , dir(immaturePoint.dir)
      , stddev(immaturePoint.stddev)
      , minDepth(immaturePoint.minDepth)
      , maxDepth(immaturePoint.maxDepth) {
    activate(immaturePoint.depth);
  }

  inline void activate(double depth) {
    state = ACTIVE;
    logInvDepth = -std::log(depth);
  }

  inline double depth() const { return std::exp(-logInvDepth); }

  Vec2 p;
  Vec3 dir;
  double logInvDepth;
  double stddev;
  double minDepth, maxDepth;
  State state;
};

} // namespace mdso

#endif
