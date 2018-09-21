#pragma once

#include "util/types.h"
#include <Eigen/Core>
#include <opencv2/core.hpp>

namespace fishdso {

struct InterestPoint {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  InterestPoint(Vec2 p, double depth = -1, double variance = 1)
      : p(p), depth(depth), variance(variance) {}

  Vec2 p;
  double depth;
  double variance;
};

} // namespace fishdso
