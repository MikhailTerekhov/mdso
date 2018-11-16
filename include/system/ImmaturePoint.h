#ifndef INCLUDE_IMMATURE_POINT
#define INCLUDE_IMMATURE_POINT

#include "util/types.h"

namespace fishdso {

struct ImmaturePoint {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ImmaturePoint(const Vec2 &p) : p(p) {}

  Vec2 p;
  double depth;
  double variance;
};

} // namespace fishdso

#endif
