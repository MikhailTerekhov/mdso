#include "system/MarginalizedPreKeyFrame.h"

namespace fishdso {

MarginalizedPreKeyFrame::MarginalizedPreKeyFrame(
    MarginalizedKeyFrame *baseFrame, const PreKeyFrame &preKeyFrame)
    : baseFrame(baseFrame)
    , baseToThis(preKeyFrame.baseToThis)
    , timestamp(preKeyFrame.timestamp) {
  for (int i = 0; i < preKeyFrame.cam->bundle.size(); ++i)
    lightBaseToFrame[i] = preKeyFrame.frames[i].lightBaseToThis;
}

} // namespace fishdso
