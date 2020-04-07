#ifndef INCLUDE_SETTINGS
#define INCLUDE_SETTINGS

#include "util/defs.h"
#include "util/types.h"
#include <cmath>
#include <gflags/gflags.h>

namespace fishdso {

struct InitializerSettings;

struct PointTracerSettings;

struct FrameTrackerSettings;

struct BundleAdjusterSettings;

struct Settings {
  struct CameraModel {
    static constexpr int default_mapPolyDegree = 10;
    int mapPolyDegree = default_mapPolyDegree;

    static constexpr int default_mapPolyPoints = 2000;
    int mapPolyPoints = default_mapPolyPoints;
  } cameraModel;

  struct PixelSelector {
    static constexpr int default_initialAdaptiveBlockSize = 25;
    int initialAdaptiveBlockSize = default_initialAdaptiveBlockSize;

    static constexpr int default_initialPointsFound = 1000;
    int initialPointsFound = default_initialPointsFound;

    static constexpr double default_adaptToFactor = 1.1;
    double adaptToFactor = default_adaptToFactor;

    static const std::vector<double> default_gradThresholds;
    std::vector<double> gradThresholds = default_gradThresholds;

    static const std::vector<cv::Scalar> default_pointColors;
    std::vector<cv::Scalar> pointColors = default_pointColors;
  } pixelSelector;

  struct DistanceMap {
    static constexpr double default_maxWidth = 480;
    double maxWidth = default_maxWidth;

    static constexpr double default_maxHeight = 302;
    double maxHeight = default_maxHeight;
  } distanceMap;

  struct StereoMatcher {
    struct StereoGeometryEstimator {
      static constexpr double default_outlierReprojError = 1.25;
      double outlierReprojError = default_outlierReprojError;

      static constexpr double default_successProb = 0.999;
      double successProb = default_successProb;

      static constexpr int default_maxRansacIter = 100000;
      int maxRansacIter = default_maxRansacIter;

      static constexpr int default_initialInliersCapacity = 2000;
      int initialInliersCapacity = default_initialInliersCapacity;

      static constexpr bool default_runAveraging = true;
      bool runAveraging = default_runAveraging;

      static constexpr bool default_runMaxRansacIter = false;
      bool runMaxRansacIter = default_runMaxRansacIter;

      static constexpr int minimalSolveN = 5;
    } stereoGeometryEstimator;

    static constexpr double default_matchNonMoveDist = 8.0;
    double matchNonMoveDist = default_matchNonMoveDist;

    static constexpr int default_keyPointNum = 2000;
    int keyPointNum = default_keyPointNum;

    static constexpr int default_maxRansacIter = 100000;
    int maxRansacIter = default_maxRansacIter;
  } stereoMatcher;

  struct Triangulation {
    static constexpr double default_epsPointIsOnSegment = 1e-9;
    double epsPointIsOnSegment = default_epsPointIsOnSegment;

    static constexpr double default_epsSamePoints = 1e-9;
    double epsSamePoints = default_epsSamePoints;

    static constexpr double default_drawPadding = 0.1;
    double drawPadding = default_drawPadding;
  } triangulation;

  struct DelaunayDsoInitializer {
    static constexpr int default_firstFramesSkip = 15;
    int firstFramesSkip = default_firstFramesSkip;

    static constexpr bool default_usePlainTriangulation = false;
    bool usePlainTriangulation = default_usePlainTriangulation;
  } delaunayDsoInitializer;

  struct KeyFrame {
    static constexpr int default_pointsNum = 2000;
    int pointsNum = default_pointsNum;
  } keyFrame;

  struct PointTracer {
    static constexpr int default_onImageTestCount = 100;
    int onImageTestCount = default_onImageTestCount;

    static constexpr double default_maxSearchRel = 0.027;
    double maxSearchRel = default_maxSearchRel;

    static constexpr double default_positionVariance = 3.2;
    double positionVariance = default_positionVariance;

    static constexpr double default_minSecondBestDistance = 3.0;
    double minSecondBestDistance = default_minSecondBestDistance;

    static constexpr double default_imprFactor = 1.0;
    double imprFactor = default_imprFactor;

    static constexpr double default_outlierEnergyFactor = 0.444;
    double outlierEnergyFactor = default_outlierEnergyFactor;

    static constexpr double default_secondBestEnergyThresFactor = 0.004;
    double secondBestEnergyThresFactor = default_secondBestEnergyThresFactor;

    static constexpr double default_outlierQuality = 3.0;
    double outlierQuality = default_outlierQuality;

    static constexpr double default_optimizedStddev = 2.4;
    double optimizedStddev = default_optimizedStddev;

    static constexpr int default_gnIter = 3;
    int gnIter = default_gnIter;

    static constexpr bool default_performFullTracing = false;
    bool performFullTracing = default_performFullTracing;

    static constexpr bool default_useAltHWeighting = true;
    bool useAltHWeighting = default_useAltHWeighting;
  } pointTracer;

  struct FrameTracker {
    static constexpr double default_trackFailFactor = 1.5;
    double trackFailFactor = default_trackFailFactor;

    static constexpr bool default_useGradWeighting = false;
    bool useGradWeighting = default_useGradWeighting;
  } frameTracker;

  struct BundleAdjuster {
    static constexpr int default_maxIterations = 10;
    int maxIterations = default_maxIterations;

    static constexpr bool default_fixedMotionOnFirstAdjustent = false;
    bool fixedMotionOnFirstAdjustent = default_fixedMotionOnFirstAdjustent;

    static constexpr bool default_fixedRotationOnSecondKF = false;
    bool fixedRotationOnSecondKF = default_fixedRotationOnSecondKF;

    static constexpr bool default_runBA = true;
    bool runBA = default_runBA;
  } bundleAdjuster;

  struct Pyramid {
    static constexpr int max_levelNum = 8;
    static constexpr int default_levelNum = 6;
    int levelNum = default_levelNum;
  } pyramid;

  struct AffineLight {
    static constexpr double default_minAffineLightA = -0.0953101798;
    double minAffineLightA = default_minAffineLightA;

    static constexpr double default_maxAffineLightA = 0.0953101798; // ln(1.1)
    double maxAffineLightA = default_maxAffineLightA;

    static constexpr double default_minAffineLightB = -0.1 * 256;
    double minAffineLightB = default_minAffineLightB;

    static constexpr double default_maxAffineLightB = 0.1 * 256;
    double maxAffineLightB = default_maxAffineLightB;

    static constexpr bool default_optimizeAffineLight = false;
    bool optimizeAffineLight = default_optimizeAffineLight;
  } affineLight;

  struct Depth {
    static constexpr double default_min = 1e-3;
    double min = default_min;

    static constexpr double default_max = 1e4;
    double max = default_max;
  } depth;

  struct GradWeighting {
    static constexpr double default_c = 50.0;
    double c = default_c;
  } gradWeighting;

  struct ResidualPattern {
    ResidualPattern(const StdVector<Vec2> &newPattern = default_pattern)
        : _pattern(newPattern) {
      height =
          int(std::ceil(std::max_element(_pattern.begin(), _pattern.end(),
                                         [](const Vec2 &a, const Vec2 &b) {
                                           return a.lpNorm<Eigen::Infinity>() <
                                                  b.lpNorm<Eigen::Infinity>();
                                         })
                            ->lpNorm<Eigen::Infinity>()));
    }

    inline const StdVector<Vec2> &pattern() const { return _pattern; }
    int height;

  private:
    StdVector<Vec2> _pattern;
    static const StdVector<Vec2> default_pattern;
  } residualPattern;

  struct Intencity {
    static constexpr double default_outlierDiff = 12.0;
    double outlierDiff = default_outlierDiff;
  } intencity;

  struct Threading {
    static constexpr int default_numThreads = 4;
    int numThreads = default_numThreads;
  } threading;

  static constexpr int default_maxOptimizedPoints = 2000;
  int maxOptimizedPoints = default_maxOptimizedPoints;

  static constexpr int default_maxKeyFrames = 6;
  int maxKeyFrames = default_maxKeyFrames;

  static constexpr bool default_trackFromLastKf = true;
  bool trackFromLastKf = default_trackFromLastKf;

  static constexpr bool default_predictUsingScrew = false;
  bool predictUsingScrew = default_predictUsingScrew;

  static constexpr bool default_continueChoosingKeyFrames = true;
  bool continueChoosingKeyFrames = default_continueChoosingKeyFrames;

  static constexpr int default_initialMaxFrame = 2500;
  int initialMaxFrame = default_initialMaxFrame;

  static constexpr int default_shiftBetweenKeyFrames = 10;
  int shiftBetweenKeyFrames = default_shiftBetweenKeyFrames;

  InitializerSettings getInitializerSettings() const;
  PointTracerSettings getPointTracerSettings() const;
  FrameTrackerSettings getFrameTrackerSettings() const;
  BundleAdjusterSettings getBundleAdjusterSettings() const;
};

struct PointTracerSettings {
  Settings::PointTracer pointTracer = {};
  Settings::Intencity intencity = {};
  Settings::ResidualPattern residualPattern = {};
  Settings::Pyramid pyramid = {};
};

struct InitializerSettings {
  Settings::DelaunayDsoInitializer initializer = {};
  Settings::StereoMatcher stereoMatcher = {};
  Settings::Threading threading = {};
  Settings::Triangulation triangulation = {};
  Settings::KeyFrame keyFrame = {};
  PointTracerSettings tracingSettings = {};
};

struct FrameTrackerSettings {
  Settings::FrameTracker frameTracker = {};
  Settings::Pyramid pyramid = {};
  Settings::AffineLight affineLight = {};
  Settings::Intencity intencity = {};
  Settings::GradWeighting gradWeighting = {};
  Settings::Threading threading = {};
};

struct BundleAdjusterSettings {
  Settings::BundleAdjuster bundleAdjuster = {};
  Settings::ResidualPattern residualPattern = {};
  Settings::GradWeighting gradWeighting = {};
  Settings::Intencity intencity = {};
  Settings::AffineLight affineLight = {};
  Settings::Threading threading = {};
  Settings::Depth depth = {};
};

} // namespace fishdso

#endif
