#ifndef INCLUDE_IMMATUREPOINT
#define INCLUDE_IMMATUREPOINT

#include "system/AffineLightTransform.h"
#include "system/CameraModel.h"
#include "system/PreKeyFrame.h"
#include "util/settings.h"
#include "util/types.h"

namespace fishdso {

struct ImmaturePoint {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  enum State { ACTIVE, OOB, OUTLIER };
  enum TracingDebugType { NO_DEBUG, DRAW_EPIPOLE };

  ImmaturePoint(PreKeyFrame *baseFrame, const Vec2 &p);

  void pointsToTrace(const SE3 &baseToRef, StdVector<Vec2> &points,
                     std::vector<Vec3> &directions);
  void traceOn(const PreKeyFrame &refFrame, TracingDebugType debugType);

  static void
  drawTracing(cv::Mat &frame,
              const StdVector<std::pair<Vec2, double>> &energiesFound,
              int lineWidth);

  Vec2 p;
  Vec3 baseDirections[settingResidualPatternSize];
  double baseIntencities[settingResidualPatternSize];
  double minDepth, maxDepth;
  double depth;
  double bestQuality;
  double lastEnergy;
  const PreKeyFrame *baseFrame;
  CameraModel *cam;
  State state;
};

} // namespace fishdso

#endif
