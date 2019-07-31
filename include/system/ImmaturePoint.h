#ifndef INCLUDE_IMMATUREPOINT
#define INCLUDE_IMMATUREPOINT

#include "system/AffineLightTransform.h"
#include "system/CameraModel.h"
#include "system/PreKeyFrame.h"
#include "util/settings.h"
#include "util/types.h"

namespace fishdso {

struct KeyFrame;

struct ImmaturePoint {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  enum State { ACTIVE, OOB, OUTLIER };
  enum TracingDebugType { NO_DEBUG, DRAW_EPIPOLE };
  enum TracingStatus {
    OK,
    WAS_OOB,
    BIG_PREDICTED_ERROR,
    EPIPOLAR_OOB,
    TOO_COARSE_PYR_LEVEL,
    INF_DEPTH,
    INF_ENERGY,
    BIG_ENERGY,
    SMALL_ABS_SECOND_BEST,
    LOW_QUALITY
  };

  // TODO create PointTracer!!!
  ImmaturePoint(KeyFrame *baseFrame, const Vec2 &p,
                const PointTracerSettings &_settings = {});

  TracingStatus traceOn(const KeyFrame &baseFrame, const PreKeyFrame &refFrame,
                        TracingDebugType debugType);

  static void
  drawTracing(cv::Mat &frame,
              const StdVector<std::pair<Vec2, double>> &energiesFound,
              int lineWidth);

  bool isReady(); // checks if the point is good enough to be optimized

  Vec2 p;
  std::vector<Vec3> baseDirections;
  std::vector<double> baseIntencities;
  StdVector<Vec2> baseGrad;
  StdVector<Vec2> baseGradNorm;
  double minDepth, maxDepth;
  double depth;
  double bestQuality;
  double lastEnergy;
  double stddev; // predicted disparity error on last successful tracing
  CameraModel *cam;
  State state;

  PointTracerSettings settings;

  // output only
  bool lastTraced;
  int numTraced;
  double depthBeforeSubpixel;
  double lastIntVar, lastGeomVar, lastFullVar;
  int tracedPyrLevel;
  bool pyrChanged;
  double eBeforeSubpixel, eAfterSubpixel;

private:
  bool pointsToTrace(const SE3 &baseToRef, Vec3 &dirMinDepth, Vec3 &dirMaxDepth,
                     StdVector<Vec2> &points, std::vector<Vec3> &directions);
  double estVariance(const Vec2 &searchDirection);
  Vec2 tracePrecise(
      const ceres::BiCubicInterpolator<ceres::Grid2D<unsigned char, 1>>
          &refFrame,
      const Vec2 &from, const Vec2 &to, const std::vector<double> &intencities,
      const StdVector<Vec2> &pattern, double &bestDispl, double &bestEnergy);
};

} // namespace fishdso

#endif
