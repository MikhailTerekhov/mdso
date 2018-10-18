#ifndef INCLUDE_INTERESTPOINT
#define INCLUDE_INTERESTPOINT

#include "util/types.h"
#include <Eigen/Core>
#include <Eigen/StdVector>
#include <opencv2/core.hpp>

namespace fishdso {

struct InterestPoint {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  enum State { ACTIVE, OUTLIER };

  InterestPoint(const Vec2 &p, double invDepth = -1, double variance = 1)
      : p(p), invDepth(invDepth), variance(variance), state(ACTIVE) {}

  Vec2 p;
  double invDepth;
  double variance;
  State state;
};

} // namespace fishdso

#endif
