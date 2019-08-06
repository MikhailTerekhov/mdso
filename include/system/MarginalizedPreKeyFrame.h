#ifndef INCLUDE_MARGINALIZEDPREKEYFRAME
#define INCLUDE_MARGINALIZEDPREKEYFRAME

#include "system/AffineLightTransform.h"
#include "system/PreKeyFrame.h"
#include "util/settings.h"
#include "util/types.h"

namespace fishdso {

struct MarginalizedKeyFrame;

struct MarginalizedPreKeyFrame {
  MarginalizedPreKeyFrame(MarginalizedKeyFrame *baseFrame,
                          const PreKeyFrame &preKeyFrame);

  MarginalizedKeyFrame *baseFrame;
  SE3 baseToThis;
  AffLight lightBaseToFrame[Settings::CameraBundle::max_camerasInBundle];
  long long timestamp;
};

} // namespace fishdso

#endif
