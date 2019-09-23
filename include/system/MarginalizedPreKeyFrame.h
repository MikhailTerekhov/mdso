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

    Timestamp timestamp;
    AffLight lightBaseToThis;
  };

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  MarginalizedPreKeyFrame(MarginalizedKeyFrame *baseFrame,
                          const PreKeyFrame &preKeyFrame);

  MarginalizedKeyFrame *baseFrame;
  SE3 baseToThis;
  std::vector<FrameEntry> frames;
  Timestamp timestamp;
};

} // namespace fishdso

#endif
