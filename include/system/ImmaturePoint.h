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

  void traceOn(const PreKeyFrame &refFrame, TracingDebugType debugType);

  static void
  drawTracing(cv::Mat &frame,
              const StdVector<std::pair<Vec2, double>> &energiesFound,
              int lineWidth);

  Vec2 p;
  Vec3 baseDirections[settingResidualPatternSize];
  double baseIntencities[settingResidualPatternSize];
  Vec2 baseGrad[settingResidualPatternSize];
  Vec2 baseGradNorm[settingResidualPatternSize];
  double minDepth, maxDepth;
  double depth;
  double bestQuality;
  double lastEnergy;
  const PreKeyFrame *baseFrame;
  CameraModel *cam;
  State state;

  // output only
  double lastIntVar, lastGeomVar, lastFullVar;

private:
  bool pointsToTrace(const SE3 &baseToRef, Vec3 &dirMinDepth, Vec3 &dirMaxDepth,
                     StdVector<Vec2> &points, std::vector<Vec3> &directions);
  double estVariance(const Vec2 &searchDirection);
};

} // namespace fishdso

#endif
