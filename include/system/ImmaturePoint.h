#ifndef INCLUDE_IMMATUREPOINT
#define INCLUDE_IMMATUREPOINT

#include "system/AffineLightTransform.h"
#include "system/CameraModel.h"
#include "system/PreKeyFrame.h"
#include "util/settings.h"
#include "util/types.h"

namespace fishdso {

struct KeyFrameEntry;
struct KeyFrame;

struct ImmaturePoint {
  static constexpr int MS = Settings::ResidualPattern::max_size;

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
    LOW_QUALITY,
    STATUS_COUNT
  };

  static std::string statusName(ImmaturePoint::TracingStatus status);

  // TODO create PointTracer!!!
  ImmaturePoint(KeyFrameEntry *host, const Vec2 &p,
                const PointTracerSettings &settings = {});

  TracingStatus traceOn(const PreKeyFrame::FrameEntry &refFrameEntry,
                        TracingDebugType debugType,
                        const PointTracerSettings &settings);

  static void drawTracing(cv::Mat &frame,
                          std::pair<Vec2, double> energiesFound[], int size,
                          int lineWidth);

  bool isReady(); // checks if the point is good enough to be optimized

  Vec2 p;
  Vec3 baseDirections[MS];
  Vec3 dir;
  double baseIntencities[MS];
  Vec2 baseGrad[MS];
  Vec2 baseGradNorm[MS];
  double minDepth, maxDepth;
  double depth;
  double bestQuality;
  double lastEnergy;
  double stddev; // predicted disparity error on last successful tracing
  CameraBundle *cam;
  CameraModel *camBase;
  State state;
  KeyFrameEntry *host;
  bool mIsReady;

  // output only
  bool lastTraced;
  int numTraced;
  double depthBeforeSubpixel;
  double lastVar;
  int tracedPyrLevel;
  bool pyrChanged;
  double eBeforeSubpixel, eAfterSubpixel;

private:
  int pointsToTrace(const CameraModel &camRef, const SE3 &baseToRef,
                    Vec3 &dirMinDepth, Vec3 &dirMaxDepth, Vec2 points[],
                    Vec3 directions[], const PointTracerSettings &settings);
  double estVariance(const Vec2 &searchDirection, const PointTracerSettings &settings);
  Vec2
  tracePrecise(const ceres::BiCubicInterpolator<ceres::Grid2D<unsigned char, 1>>
                   &refFrame,
               const Vec2 &from, const Vec2 &to, double intencities[],
               Vec2 pattern[], double &bestDispl, double &bestEnergy,
               const PointTracerSettings &settings);
};

} // namespace fishdso

#endif
