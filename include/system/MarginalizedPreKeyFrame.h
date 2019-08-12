#ifndef INCLUDE_MARGINALIZEDPREKEYFRAME
#define INCLUDE_MARGINALIZEDPREKEYFRAME

#include "system/AffineLightTransform.h"
#include "system/PreKeyFrame.h"
#include "util/settings.h"
#include "util/types.h"

namespace fishdso {

struct MarginalizedKeyFrame;

struct MarginalizedPreKeyFrame {

  struct FrameEntry {
    FrameEntry();
    FrameEntry(const PreKeyFrame::FrameEntry &entry);

    long long timestamp;
    AffLight lightBaseToThis;
  };

  MarginalizedPreKeyFrame(MarginalizedKeyFrame *baseFrame,
                          const PreKeyFrame &preKeyFrame);

  MarginalizedKeyFrame *baseFrame;
  SE3 baseToThis;
  FrameEntry frames[Settings::CameraBundle::max_camerasInBundle];
  long long timestamp;
};

} // namespace fishdso

#endif
