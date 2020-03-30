#include "system/OptimizedPoint.h"
#include "system/serialization.h"

namespace mdso {

OptimizedPoint::OptimizedPoint(const ImmaturePoint &immaturePoint)
    : p(immaturePoint.p)
    , dir(immaturePoint.dir)
    , stddev(immaturePoint.stddev)
    , minDepth(immaturePoint.minDepth)
    , maxDepth(immaturePoint.maxDepth) {
  activate(immaturePoint.depth);
}

OptimizedPoint::OptimizedPoint(KeyFrameEntry *,
                               PointSerializer<LOAD> &pointSerializer) {
  pointSerializer.process(*this);
}

void OptimizedPoint::activate(double depth) {
  state = ACTIVE;
  logDepth = std::log(depth);
}

} // namespace mdso