#ifndef INCLUDE_TRACKINGPREDICTOR
#define INCLUDE_TRACKINGPREDICTOR

#include "system/FrameTracker.h"
#include "util/TrajectoryHolder.h"
#include "util/types.h"

namespace mdso {

class TrackingPredictor {
public:
  virtual ~TrackingPredictor();

  TrackingResult predictAt(Timestamp timestamp, int baseInd) const;

protected:
  virtual TrajectoryHolder *trajectoryHolder() const = 0;
  virtual SE3 predictBodyToWorldAt(Timestamp timestamp) const = 0;
  virtual AffLight predictAffLightWorldToBodyAt(Timestamp timestamp,
                                                int camInd) const;
};

} // namespace mdso

#endif