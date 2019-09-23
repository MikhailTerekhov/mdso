#include "system/MarginalizedPreKeyFrame.h"

namespace mdso {

MarginalizedPreKeyFrame::FrameEntry::FrameEntry()
    : timestamp(0) {}

MarginalizedPreKeyFrame::FrameEntry::FrameEntry(
    const PreKeyFrame::FrameEntry &entry)
    : timestamp(entry.timestamp)
    , lightBaseToThis(entry.lightBaseToThis) {}

MarginalizedPreKeyFrame::MarginalizedPreKeyFrame(
    MarginalizedKeyFrame *baseFrame, const PreKeyFrame &preKeyFrame)
    : baseFrame(baseFrame) {
  frames.reserve(preKeyFrame.cam->bundle.size());
  for (int i = 0; i < preKeyFrame.cam->bundle.size(); ++i)
    frames.emplace_back(preKeyFrame.frames[i]);
}

} // namespace mdso
