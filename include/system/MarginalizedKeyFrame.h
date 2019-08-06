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
    AffLight lightWorldToThis;
  };

  MarginalizedKeyFrame(const KeyFrame &keyFrame);

  static_vector<FrameEntry, Settings::CameraBundle::max_camerasInBundle> frames;
  static_vector<std::unique_ptr<MarginalizedPreKeyFrame>,
                Settings::max_keyFrameDist>
      trackedFrames;
  SE3 thisToWorld;
  long long timestamp;
};

} // namespace fishdso

#endif
