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

  enum State { ACTIVE, OOB };
  enum TracingDebugType { NO_DEBUG, DRAW_EPIPOLE };

  ImmaturePoint(PreKeyFrame *baseFrame, const Vec2 &p);

  void traceOn(const PreKeyFrame &refFrame, TracingDebugType debugType);

  static void
  drawTracing(cv::Mat &frame,
              const StdVector<std::pair<Vec2, double>> &energiesFound, int lineWidth);

  Vec2 p;
  Vec3 baseDirections[settingResidualPatternSize];
  double baseIntencities[settingResidualPatternSize];
  double minDepth, maxDepth;
  double depth;
  double quality;
  const PreKeyFrame *baseFrame;
  CameraModel *cam;
  State state;
};

} // namespace fishdso

#endif
