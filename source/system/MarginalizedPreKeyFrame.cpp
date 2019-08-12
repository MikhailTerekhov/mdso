#include "system/MarginalizedPreKeyFrame.h"

namespace fishdso {

MarginalizedPreKeyFrame::FrameEntry::FrameEntry()
    : timestamp(0) {}

MarginalizedPreKeyFrame::FrameEntry::FrameEntry(
    const PreKeyFrame::FrameEntry &entry)
    : timestamp(entry.timestamp)
    , lightBaseToThis(entry.lightBaseToThis) {}

MarginalizedPreKeyFrame::MarginalizedPreKeyFrame(
    MarginalizedKeyFrame *baseFrame, const PreKeyFrame &preKeyFrame)
    : baseFrame(baseFrame) {
  for (int i = 0; i < preKeyFrame.cam->bundle.size(); ++i)
    frames[i] = FrameEntry(preKeyFrame.frames[i]);
}

} // namespace fishdso
