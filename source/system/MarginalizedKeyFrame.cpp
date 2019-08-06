#include "system/MarginalizedKeyFrame.h"

namespace fishdso {

MarginalizedKeyFrame::FrameEntry::Point::Point(const ImmaturePoint &ip)
    : point(ip.depth * ip.baseDirections[0])
    , minDepth(ip.minDepth)
    , maxDepth(ip.maxDepth) {}

MarginalizedKeyFrame::FrameEntry::Point::Point(const OptimizedPoint &op)
    : point(op.depth() * op.dir)
    , minDepth(op.minDepth)
    , maxDepth(op.maxDepth) {}

MarginalizedKeyFrame::FrameEntry::FrameEntry(const KeyFrameEntry &entry)
    : lightWorldToThis(entry.lightWorldToThis) {
  for (const ImmaturePoint &ip : entry.immaturePoints)
    points.emplace_back(ip);
  for (const OptimizedPoint &op : entry.optimizedPoints)
    points.emplace_back(op);
}

MarginalizedKeyFrame::MarginalizedKeyFrame(const KeyFrame &keyFrame)
    : thisToWorld(keyFrame.thisToWorld)
    , timestamp(keyFrame.preKeyFrame->timestamp) {
  for (int i = 0; i < keyFrame.preKeyFrame->cam->bundle.size(); ++i)
    frames.emplace_back(keyFrame.frames[i]);
  for (const auto &pkf : keyFrame.trackedFrames)
    trackedFrames.emplace_back(this, pkf);
}

} // namespace fishdso
