#ifndef INCLUDE_TRACKINGPREDICTORROT
#define INCLUDE_TRACKINGPREDICTORROT

#include "system/TrackingPredictor.h"

namespace mdso {

class TrackingPredictorRot : public TrackingPredictor {
public:
  TrackingPredictorRot(TrajectoryHolder *_trajectoryHolder);

private:
  TrajectoryHolder *trajectoryHolder() const override;
  SE3 predictBodyToWorldAt(Timestamp timestamp) const override;

  TrajectoryHolder *mTrajectoryHolder;
};

} // namespace mdso

#endif
