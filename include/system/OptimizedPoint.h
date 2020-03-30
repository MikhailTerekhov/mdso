#ifndef INCLUDE_INTERESTPOINT
#define INCLUDE_INTERESTPOINT

#include "system/ImmaturePoint.h"
#include "system/SerializerMode.h"
#include "util/types.h"
#include <Eigen/Core>
#include <Eigen/StdVector>
#include <cmath>
#include <opencv2/core.hpp>

namespace mdso {

template <SerializerMode mode> class PointSerializer;

struct OptimizedPoint {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  enum State { ACTIVE, OOB, OUTLIER };

  OptimizedPoint(const ImmaturePoint &immaturePoint);
  OptimizedPoint(KeyFrameEntry *, PointSerializer<LOAD> &pointSerializer);

  void activate(double depth);

  inline double depth() const { return std::exp(logDepth); }

  Vec2 p;
  Vec3 dir;
  double logDepth;
  double stddev;
  double minDepth, maxDepth;
  State state;
};

} // namespace mdso

#endif
