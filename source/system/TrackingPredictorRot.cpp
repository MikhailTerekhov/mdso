#include "system/TrackingPredictorRot.h"

namespace mdso {

TrackingPredictorRot::TrackingPredictorRot(TrajectoryHolder *_trajectoryHolder)
    : mTrajectoryHolder(_trajectoryHolder) {}

TrajectoryHolder *TrackingPredictorRot::trajectoryHolder() const {
  return mTrajectoryHolder;
}

SE3 TrackingPredictorRot::predictBodyToWorldAt(
    mdso::Timestamp timestamp) const {
  int size = trajectoryHolder()->trajectorySize();
  CHECK_GT(size, 0);
  if (size == 1)
    return trajectoryHolder()->bodyToWorld(0);

  Timestamp lastTs = trajectoryHolder()->timestamp(size - 1);
  Timestamp lastButOneTs = trajectoryHolder()->timestamp(size - 2);
  CHECK_LT(lastButOneTs, lastTs);
  double timeLastByLbo = double(timestamp - lastTs) / (lastTs - lastButOneTs);

  SE3 lastToWorld = trajectoryHolder()->bodyToWorld(size - 1);
  SE3 lastButOneToWorld = trajectoryHolder()->bodyToWorld(size - 2);
  SE3 lboToLast = lastToWorld.inverse() * lastButOneToWorld;
  SO3 lastToCurRot = SO3::exp(timeLastByLbo * lboToLast.so3().log());
  Vec3 lastToCurTrans =
      timeLastByLbo *
      (lastToCurRot * lboToLast.so3().inverse() * lboToLast.translation());
  return lastToWorld * SE3(lastToCurRot, lastToCurTrans).inverse();
}

} // namespace mdso
