#include "system/TrackingPredictor.h"

namespace mdso {

TrackingPredictor::~TrackingPredictor() {}

TrackingResult TrackingPredictor::predictAt(Timestamp timestamp,
                                            int baseInd) const {
  TrajectoryHolder *traj = trajectoryHolder();
  int camNumber = traj->camNumber();
  int size = traj->trajectorySize();
  CHECK(baseInd >= 0 && baseInd <= size);

  SE3 trackedToWorld = predictBodyToWorldAt(timestamp);
  SE3 baseToWorld = traj->bodyToWorld(baseInd);
  TrackingResult predicted(camNumber);
  predicted.baseToTracked = trackedToWorld.inverse() * baseToWorld;
  for (int camInd = 0; camInd < camNumber; ++camInd) {
    AffLight affLightWorldToTracked =
        predictAffLightWorldToBodyAt(timestamp, camInd);
    AffLight affLightWorldToBase = traj->affLightWorldToBody(baseInd, camInd);
    predicted.lightBaseToTracked[camInd] =
        affLightWorldToTracked * affLightWorldToBase.inverse();
  }
  return predicted;
}

AffLight TrackingPredictor::predictAffLightWorldToBodyAt(Timestamp timestamp,
                                                         int camInd) const {
  TrajectoryHolder *traj = trajectoryHolder();
  int size = traj->trajectorySize();
  CHECK(size > 0);
  return traj->affLightWorldToBody(size - 1, camInd);
}

} // namespace mdso
