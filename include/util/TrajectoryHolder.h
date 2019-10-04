#ifndef INCLUDE_TRAJECTORY_HOLDER
#define INCLUDE_TRAJECTORY_HOLDER

#include "util/types.h"

namespace mdso {

class TrajectoryHolder {
public:
  virtual ~TrajectoryHolder();

  virtual int trajectorySize() const = 0;
  virtual int camNumber() const = 0;
  virtual Timestamp timestamp(int ind) const = 0;
  virtual SE3 bodyToWorld(int ind) const = 0;
  virtual AffLight affLightWorldToBody(int ind, int camInd) const = 0;
};
} // namespace mdso

#endif