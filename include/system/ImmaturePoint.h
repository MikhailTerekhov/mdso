#ifndef INCLUDE_IMMATUREPOINT
#define INCLUDE_IMMATUREPOINT

#include "util/types.h"
#include "util/settings.h"
#include "system/CameraModel.h"
#include "system/AffineLightTransform.h"
#include "system/PreKeyFrame.h"

namespace fishdso {

struct ImmaturePoint {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
    enum State { ACTIVE, OOB };

  ImmaturePoint(PreKeyFrame *baseFrame, const Vec2 &p);

  void traceOn(const PreKeyFrame &refFrame, bool debugOut);

  Vec2 p;
  Vec3 baseDirections[settingResidualPatternSize];
  double baseIntencities[settingResidualPatternSize];
  double minDepth, maxDepth;
  double depth;
  double quality;
  const PreKeyFrame *baseFrame;
  CameraModel *cam;
  State state;
};

} // namespace fishdso

#endif
