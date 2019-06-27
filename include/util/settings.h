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
    CameraModel()
        : mapPolyDegree(default_mapPolyDegree)
        , mapPolyPoints(default_mapPolyPoints) {}

    int mapPolyDegree;
    static constexpr int default_mapPolyDegree = 10;

    int mapPolyPoints;
    static constexpr int default_mapPolyPoints = 2000;
  } cameraModel;

  struct PixelSelector {
    PixelSelector()
        : initialAdaptiveBlockSize(default_initialAdaptiveBlockSize)
        , initialPointsFound(default_initialPointsFound)
        , adaptToFactor(default_adaptToFactor)
        , gradThresholds(default_gradThresholds)
        , pointColors(default_pointColors) {}

    int initialAdaptiveBlockSize;
    static constexpr int default_initialAdaptiveBlockSize = 25;

    int initialPointsFound;
    static constexpr int default_initialPointsFound = 1000;

    double adaptToFactor;
    static constexpr double default_adaptToFactor = 1.1;

    std::vector<double> gradThresholds;
    static const std::vector<double> default_gradThresholds;

    std::vector<cv::Scalar> pointColors;
    static const std::vector<cv::Scalar> default_pointColors;
  } pixelSelector;

  struct DistanceMap {
    DistanceMap()
        : maxWidth(default_maxWidth)
        , maxHeight(default_maxHeight) {}

    double maxWidth;
    static constexpr double default_maxWidth = 480;

    double maxHeight;
    static constexpr double default_maxHeight = 302;
  } distanceMap;

  struct StereoMatcher {
    StereoMatcher()
        : matchNonMoveDist(default_matchNonMoveDist)
        , keyPointNum(default_keyPointNum) {}

    struct StereoGeometryEstimator {
      StereoGeometryEstimator()
          : outlierReprojError(default_outlierReprojError)
          , successProb(default_successProb)
          , maxRansacIter(default_maxRansacIter)
          , initialInliersCapacity(default_initialInliersCapacity)
          , runAveraging(default_runAveraging)
          , runMaxRansacIter(default_runMaxRansacIter) {}

      double outlierReprojError;
      static constexpr double default_outlierReprojError = 1.25;

      double successProb;
      static constexpr double default_successProb = 0.999;

      int maxRansacIter;
      static constexpr int default_maxRansacIter = 100000;

      int initialInliersCapacity;
      static constexpr int default_initialInliersCapacity = 2000;

      bool runAveraging;
      static constexpr bool default_runAveraging = true;

      bool runMaxRansacIter;
      static constexpr bool default_runMaxRansacIter = false;

      static constexpr int minimalSolveN = 5;
    } stereoGeometryEstimator;

    double matchNonMoveDist;
    static constexpr double default_matchNonMoveDist = 8.0;

    int keyPointNum;
    static constexpr int default_keyPointNum = 2000;

    int maxRansacIter;
    static constexpr int default_maxRansacIter = 100000;
  } stereoMatcher;

  struct Triangulation {
    Triangulation()
        : epsPointIsOnSegment(default_epsPointIsOnSegment)
        , epsSamePoints(default_epsSamePoints)
        , drawPadding(default_drawPadding) {}

    double epsPointIsOnSegment;
    static constexpr double default_epsPointIsOnSegment = 1e-9;

    double epsSamePoints;
    static constexpr double default_epsSamePoints = 1e-9;

    double drawPadding;
    static constexpr double default_drawPadding = 0.1;
  } triangulation;

  struct DelaunayDsoInitializer {
    DelaunayDsoInitializer()
        : firstFramesSkip(default_firstFramesSkip)
        , usePlainTriangulation(default_usePlainTriangulation) {}

    int firstFramesSkip;
    static constexpr int default_firstFramesSkip = 15;

    bool usePlainTriangulation;
    static constexpr bool default_usePlainTriangulation = false;
  } delaunayDsoInitializer;

  struct KeyFrame {
    KeyFrame()
        : pointsNum(default_pointsNum) {}

    int pointsNum;
    static constexpr int default_pointsNum = 2000;
  } keyFrame;

  struct PointTracer {
    PointTracer()
        : onImageTestCount(default_onImageTestCount)
        , maxSearchRel(default_maxSearchRel)
        , positionVariance(default_positionVariance)
        , minSecondBestDistance(default_minSecondBestDistance)
        , imprFactor(default_imprFactor)
        , outlierEnergyFactor(default_outlierEnergyFactor)
        , outlierQuality(default_outlierQuality)
        , optimizedStddev(default_optimizedStddev)
        , gnIter(default_gnIter)
        , performFullTracing(default_performFullTracing)
        , useAltHWeighting(default_useAltHWeighting) {}

    int onImageTestCount;
    static constexpr int default_onImageTestCount = 100;

    double maxSearchRel;
    static constexpr double default_maxSearchRel = 0.027;

    double positionVariance;
    static constexpr double default_positionVariance = 3.2;

    double minSecondBestDistance;
    static constexpr double default_minSecondBestDistance = 3.0;

    double imprFactor;
    static constexpr double default_imprFactor = 1.0;

    double outlierEnergyFactor;
    static constexpr double default_outlierEnergyFactor = 0.444;

    double outlierQuality;
    static constexpr double default_outlierQuality = 3.0;

    double optimizedStddev;
    static constexpr double default_optimizedStddev = 2.4;

    int gnIter;
    static constexpr int default_gnIter = 3;

    bool performFullTracing;
    static constexpr bool default_performFullTracing = false;

    bool useAltHWeighting;
    static constexpr bool default_useAltHWeighting = true;
  } pointTracer;

  struct FrameTracker {
    FrameTracker()
        : trackFailFactor(default_trackFailFactor)
        , useGradWeighting(default_useGradWeighting)
        , performTrackingCheckGT(default_performTrackingCheckGT)
        , performTrackingCheckStereo(default_performTrackingCheckStereo) {}

    double trackFailFactor;
    static constexpr double default_trackFailFactor = 1.5;

    bool useGradWeighting;
    static constexpr bool default_useGradWeighting = false;

    bool performTrackingCheckGT;
    static constexpr bool default_performTrackingCheckGT = false;

    bool performTrackingCheckStereo;
    static constexpr bool default_performTrackingCheckStereo = false;
  } frameTracker;

  struct BundleAdjuster {
    BundleAdjuster()
        : maxIterations(default_maxIterations)
        , fixedMotionOnFirstAdjustent(default_fixedMotionOnFirstAdjustent)
        , fixedRotationOnSecondKF(default_fixedRotationOnSecondKF)
        , runBA(default_runBA) {}

    int maxIterations;
    static constexpr int default_maxIterations = 10;

    bool fixedMotionOnFirstAdjustent;
    static constexpr bool default_fixedMotionOnFirstAdjustent = false;

    bool fixedRotationOnSecondKF;
    static constexpr bool default_fixedRotationOnSecondKF = false;

    bool runBA;
    static constexpr bool default_runBA = true;
  } bundleAdjuster;

  struct Pyramid {
    Pyramid()
        : levelNum(default_levelNum) {}

    int levelNum;
    static constexpr int default_levelNum = 6;
    static constexpr int max_levelNum = 8;
  } pyramid;

  struct AffineLight {
    AffineLight()
        : minAffineLightA(default_minAffineLightA)
        , maxAffineLightA(default_maxAffineLightA)
        , minAffineLightB(default_minAffineLightB)
        , maxAffineLightB(default_maxAffineLightB)
        , optimizeAffineLight(default_optimizeAffineLight) {}

    double minAffineLightA;
    static constexpr double default_minAffineLightA = -0.0953101798;

    double maxAffineLightA;
    static constexpr double default_maxAffineLightA = 0.0953101798; // ln(1.1)

    double minAffineLightB;
    static constexpr double default_minAffineLightB = -0.1 * 256;

    double maxAffineLightB;
    static constexpr double default_maxAffineLightB = 0.1 * 256;

    bool optimizeAffineLight;
    static constexpr bool default_optimizeAffineLight = false;
  } affineLight;

  struct Depth {
    Depth()
        : min(default_min)
        , max(default_max) {}

    double min;
    static constexpr double default_min = 1e-3;

    double max;
    static constexpr double default_max = 1e4;
  } depth;

  struct GradWeighting {
    GradWeighting()
        : c(default_c) {}

    double c;
    static constexpr double default_c = 50.0;
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
    Intencity()
        : outlierDiff(default_outlierDiff) {}

    double outlierDiff;
    static constexpr double default_outlierDiff = 12.0;
  } intencity;

  struct Threading {
    Threading()
        : numThreads(4) {}

    int numThreads;
    static constexpr int default_numThreads = 4;
  } threading;

  Settings()
      : maxOptimizedPoints(default_maxOptimizedPoints)
      , maxKeyFrames(default_maxKeyFrames)
      , trackFromLastKf(default_trackFromLastKf)
      , predictUsingScrew(default_predictUsingScrew)
      , switchFirstMotionToGT(default_switchFirstMotionToGT)
      , allPosesGT(default_allPosesGT)
      , continueChoosingKeyFrames(default_continueChoosingKeyFrames) 
      , initialMaxFrame(default_initialMaxFrame) {}

  int maxOptimizedPoints;
  static constexpr int default_maxOptimizedPoints = 2000;

  int maxKeyFrames;
  static constexpr int default_maxKeyFrames = 6;

  bool trackFromLastKf;
  static constexpr bool default_trackFromLastKf = true;

  bool predictUsingScrew;
  static constexpr bool default_predictUsingScrew = false;

  bool switchFirstMotionToGT;
  static constexpr bool default_switchFirstMotionToGT = false;

  bool allPosesGT;
  static constexpr bool default_allPosesGT = false;

  bool continueChoosingKeyFrames;
  static constexpr bool default_continueChoosingKeyFrames = true;

  int initialMaxFrame;
  static constexpr int default_initialMaxFrame = 2500;

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
  Settings::FrameTracker frameTracker;
  Settings::Pyramid pyramid;
  Settings::AffineLight affineLight;
  Settings::Intencity intencity;
  Settings::GradWeighting gradWeighting;
  Settings::Threading threading;
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
