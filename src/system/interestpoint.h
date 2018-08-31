#pragma once

#include "util/types.h"
#include <Eigen/Core>
#include <opencv2/core.hpp>

namespace fishdso {

struct InterestPoint {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Vec2 p;
  double depth;
};

} // namespace fishdso
