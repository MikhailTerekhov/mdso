#ifndef INCLUDE_MARGINALIZEDKEYFRAME
#define INCLUDE_MARGINALIZEDKEYFRAME

#include "system/KeyFrame.h"
#include "system/MarginalizedPreKeyFrame.h"
#include "util/settings.h"
#include "util/types.h"

namespace fishdso {

struct MarginalizedKeyFrame {

  struct FrameEntry {

    struct Point {
      Point(const ImmaturePoint &ip);
      Point(const OptimizedPoint &op);

      Vec3 point;
      double minDepth, maxDepth;
    };

    FrameEntry(const KeyFrameEntry &entry);

    static_vector<Point, Settings::KeyFrame::max_immaturePointsNum> points;
    Timestamp timestamp;
    AffLight lightWorldToThis;
  };

  MarginalizedKeyFrame(const KeyFrame &keyFrame);

  std::vector<FrameEntry> frames;
  std::vector<std::unique_ptr<MarginalizedPreKeyFrame>> trackedFrames;
  SE3 thisToWorld;
};

} // namespace fishdso

#endif
