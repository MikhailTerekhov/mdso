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

  static constexpr int PS = settingResidualPatternSize;
  static const int PH;     // pattern height
  static const double TH;  // outlier intencity difference threshold
  static const double SBD; // minimum second best distance squared

  enum State { ACTIVE, OOB, OUTLIER };
  enum TracingDebugType { NO_DEBUG, DRAW_EPIPOLE };

  ImmaturePoint(PreKeyFrame *baseFrame, const Vec2 &p);

  void traceOn(const PreKeyFrame &refFrame, TracingDebugType debugType);

  static void
  drawTracing(cv::Mat &frame,
              const StdVector<std::pair<Vec2, double>> &energiesFound,
              int lineWidth);

  Vec2 p;
  Vec3 baseDirections[PS];
  double baseIntencities[PS];
  Vec2 baseGrad[PS];
  Vec2 baseGradNorm[PS];
  double minDepth, maxDepth;
  double depth;
  double bestQuality;
  double lastEnergy;
  const PreKeyFrame *baseFrame;
  CameraModel *cam;
  State state;

  // output only
  double lastIntVar, lastGeomVar, lastFullVar;
  int bestPyrLevel;
  bool pyrChanged;
  double eBeforeSubpixel, eAfterSubpixel;

private:
  bool pointsToTrace(const SE3 &baseToRef, Vec3 &dirMinDepth, Vec3 &dirMaxDepth,
                     StdVector<Vec2> &points, std::vector<Vec3> &directions);
  double estVariance(const Vec2 &searchDirection);
  Vec2
  tracePrecise(const ceres::BiCubicInterpolator<ceres::Grid2D<unsigned char, 1>>
                   &refFrame,
               const Vec2 &from, const Vec2 &to, double intencities[PS],
               Vec2 pattern[PS], double &bestDispl, double &bestEnergy);
};

} // namespace fishdso

#endif
