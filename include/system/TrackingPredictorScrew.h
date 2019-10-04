#ifndef INCLUDE_TRACKINGPREDICTORSCREW
#define INCLUDE_TRACKINGPREDICTORSCREW

#include "system/TrackingPredictor.h"

namespace mdso {

class TrackingPredictorScrew : public TrackingPredictor {
public:
  TrackingPredictorScrew(TrajectoryHolder *_trajectoryHolder);

private:
  TrajectoryHolder *trajectoryHolder() const override;
  SE3 predictBodyToWorldAt(Timestamp timestamp) const override;

  TrajectoryHolder *mTrajectoryHolder;
};

} // namespace mdso

#endif