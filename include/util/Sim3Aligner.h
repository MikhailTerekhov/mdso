#ifndef INCLUDE_SIM3ALIGNER
#define INCLUDE_SIM3ALIGNER

#include "util/types.h"

namespace mdso {

class Sim3Aligner {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Sim3Aligner(const SE3 &worldToFirst, const SE3 &worldToSecond,
              const SE3 &worldToFirstGT, const SE3 &worldToSecondGT);

  SE3 alignWorldToFrameGT(const SE3 &worldToFrameGT) const;
  Vec3 alignScale(const Vec3 &pointInFrameGT) const;

private:
  double scaleGTToDso;
  SE3 gtToDso;
};

} // namespace mdso

#endif
