#ifndef INCLUDE_PRECOMPUTATIONS
#define INCLUDE_PRECOMPUTATIONS

#include "optimize/MotionDerivatives.h"
#include "optimize/Parameters.h"

namespace mdso::optimize {

class PrecomputedHostToTarget {
public:
  PrecomputedHostToTarget(CameraBundle *cam, const Parameters *parameters);

  SE3t get(int hostInd, int hostCamInd, int targetInd, int targetCamInd);

private:
  const Parameters *parameters;
  StdVector<SE3t> camToBody;
  StdVector<SE3t> bodyToCam;
  Array4d<SE3t> hostToTarget;
};

class PrecomputedMotionDerivatives {
public:
  PrecomputedMotionDerivatives(CameraBundle *cam, const Parameters *parameters);
  const MotionDerivatives &get(int hostInd, int hostCamInd, int targetInd,
                               int targetCamInd);

private:
  const Parameters *parameters;
  StdVector<SE3t> camToBody;
  StdVector<SE3t> bodyToCam;
  Array4d<std::optional<MotionDerivatives>> hostToTargetDiff;
};

class PrecomputedLightHostToTarget {
public:
  PrecomputedLightHostToTarget(const Parameters *parameters);

  AffLightT get(int hostInd, int hostCamInd, int targetInd, int targetCamInd);

private:
  const Parameters *parameters;
  Array4d<std::optional<AffLightT>> lightHostToTarget;
};

} // namespace mdso::optimize

#endif
