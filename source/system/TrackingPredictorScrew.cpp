#include "system/TrackingPredictorScrew.h"

namespace mdso {

TrackingPredictorScrew::TrackingPredictorScrew(
    TrajectoryHolder *_trajectoryHolder)
    : mTrajectoryHolder(_trajectoryHolder) {}

TrajectoryHolder *TrackingPredictorScrew::trajectoryHolder() const {
  return mTrajectoryHolder;
}

SE3 TrackingPredictorScrew::predictBodyToWorldAt(Timestamp timestamp) const {
  int size = trajectoryHolder()->trajectorySize();
  CHECK_GT(size, 0);
  if (size == 1)
    return trajectoryHolder()->bodyToWorld(0);

  Timestamp lastTs = trajectoryHolder()->timestamp(size - 1);
  Timestamp lastButOneTs = trajectoryHolder()->timestamp(size - 2);
  CHECK_LT(lastButOneTs, lastTs);
  double timeLastByLbo = double(timestamp - lastTs) / (lastTs - lastButOneTs);

  VLOG(1) << "timeLastByLbo = " << timeLastByLbo;
  VLOG(1) << "lbo ts = " << lastButOneTs << ", last ts = " << lastTs
          << ", cur ts = " << timestamp;
  VLOG(1) << "last - lbo = " << lastTs - lastButOneTs << ", cur - last"
          << timestamp - lastTs;

  SE3 lastToWorld = trajectoryHolder()->bodyToWorld(size - 1);
  SE3 lastButOneToWorld = trajectoryHolder()->bodyToWorld(size - 2);
  SE3 lboToLast = lastToWorld.inverse() * lastButOneToWorld;
  return lastToWorld * SE3::exp(timeLastByLbo * lboToLast.log()).inverse();
}

} // namespace mdso