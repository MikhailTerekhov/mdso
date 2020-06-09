#ifndef INCLUDE_SETTINGS
#define INCLUDE_SETTINGS

#include "util/defs.h"
#include "util/types.h"
#include <cmath>
#include <gflags/gflags.h>
#include <glog/logging.h>

namespace mdso {

struct InitializerDelaunaySettings;
struct InitializerGroundTruthSettings;
struct PointTracerSettings;
struct FrameTrackerSettings;
struct BundleAdjusterSettings;
struct ResidualSettings;
struct EnergyFunctionSettings;

struct Settings {
  struct CameraModel {
    static constexpr int default_mapPolyDegree = 10;
    int mapPolyDegree = default_mapPolyDegree;

    static constexpr int default_unmapPolyDegree = 6;
    int unmapPolyDegree = default_unmapPolyDegree;

    static constexpr int default_mapPolyPoints = 2000;
    int mapPolyPoints = default_mapPolyPoints;

    static constexpr int default_unmapPolyPoints = 2000;
    int unmapPolyPoints = default_unmapPolyPoints;

    static constexpr int default_magicMaxAngle = 100.0 * (M_PI / 180);
    int magicMaxAngle = default_magicMaxAngle;

    enum CenterShift { NO_SHIFT, MINUS_05, PLUS_05 };
    static constexpr CenterShift default_centerShift = NO_SHIFT;
    CenterShift centerShift = default_centerShift;
  } cameraModel;

  struct PixelSelector {
    static constexpr int max_points = 10000;

    static constexpr int default_initialAdaptiveBlockSize = 25;
    int initialAdaptiveBlockSize = default_initialAdaptiveBlockSize;

    static constexpr int default_initialPointsFound = 1000;
    int initialPointsFound = default_initialPointsFound;

    static constexpr double default_adaptToFactor = 1.1;
    double adaptToFactor = default_adaptToFactor;

    static constexpr int gradThesholdCount = 3;
    static const std::array<double, gradThesholdCount> default_gradThresholds;
    std::array<double, gradThesholdCount> gradThresholds =
        default_gradThresholds;

    static const std::array<cv::Scalar, gradThesholdCount> default_pointColors;
    std::array<cv::Scalar, gradThesholdCount> pointColors = default_pointColors;

    static constexpr double default_relDebugPointRadius = 5e-3;
    double relDebugPointRadius = default_relDebugPointRadius;
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

  struct DsoInitializer {
    static constexpr int max_initializedFrames = 2;

    static constexpr int default_firstFramesSkip = 15;
    int firstFramesSkip = default_firstFramesSkip;
  } dsoInitializer;

  struct DelaunayDsoInitializer {
    static constexpr bool default_usePlainTriangulation = false;
    bool usePlainTriangulation = default_usePlainTriangulation;
  } delaunayDsoInitializer;

  struct KeyFrame {
    static constexpr int default_immaturePointsNum = 2000;
    static constexpr int max_immaturePointsNum = 10000;
    inline int immaturePointsNum() const { return mImmaturePointsNum; }
    inline void setImmaturePointsNum(int newImmaturePointsNum) {
      CHECK(newImmaturePointsNum > 0 &&
            newImmaturePointsNum <= max_immaturePointsNum);
      mImmaturePointsNum = newImmaturePointsNum;
    }

  private:
    int mImmaturePointsNum = default_immaturePointsNum;
  } keyFrame;

  struct PointTracer {
    static constexpr int default_onImageTestCount = 100;
    int onImageTestCount = default_onImageTestCount;

    static constexpr double default_maxSearchRel = 0.027;
    double maxSearchRel = default_maxSearchRel;

    static constexpr int max_maxSearchAbs = 100;
    static constexpr int default_maxSearchAbs = 100;
    inline int maxSearchAbs() const { return mMaxSearchAbs; }
    inline void setMaxSearchAbs(int newMaxSearchAbs) {
      CHECK(newMaxSearchAbs <= max_maxSearchAbs);
      mMaxSearchAbs = newMaxSearchAbs;
    }

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

    static constexpr bool default_useAltHWeighting = true;
    bool useAltHWeighting = default_useAltHWeighting;

    static constexpr double default_relTrueDepthDelta = 0.01;
    double relTrueDepthDelta = default_relTrueDepthDelta;

  private:
    int mMaxSearchAbs = default_maxSearchAbs;
  } pointTracer;

  struct FrameTracker {
    static constexpr double default_trackFailFactor = 1.5;
    double trackFailFactor = default_trackFailFactor;

    static constexpr bool default_useGradWeighting = false;
    bool useGradWeighting = default_useGradWeighting;

    static constexpr bool default_doIntercameraReprojection = true;
    bool doIntercameraReprojection = default_doIntercameraReprojection;
  } frameTracker;

  struct Pyramid {
    static constexpr int max_levelNum = 8;
    static constexpr int default_levelNum = 6;

    inline void setLevelNum(int newLevelNum) {
      CHECK(newLevelNum > 0 && newLevelNum <= max_levelNum);
      mLevelNum = newLevelNum;
    }

    inline int levelNum() const { return mLevelNum; }

  private:
    int mLevelNum = default_levelNum;
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
    static constexpr double default_min = 1e-3; // TODO
    double min = default_min;

    static constexpr double default_max = 1e4; // TODO
    double max = default_max;

    static constexpr double default_maxAbsLogDelta = 1e1;
    double maxAbsLogDelta = default_maxAbsLogDelta;

    static constexpr bool default_setMinBound = true;
    bool setMinBound = default_setMinBound;

    static constexpr bool default_setMaxBound = true;
    bool setMaxBound = default_setMaxBound;

    static constexpr bool default_useMinPlusExpParametrization = false;
    bool useMinPlusExpParametrization = default_useMinPlusExpParametrization;
  } depth;

  struct ResidualWeighting {
    static constexpr bool default_useGradientWeighting = true;
    bool useGradientWeighting = default_useGradientWeighting;

    static constexpr double default_c = 50.0;
    double c = default_c;

    static constexpr double default_lossEps = 1e-8;
    double lossEps = default_lossEps;
  } residualWeighting;

  struct ResidualPattern {
    static constexpr int max_size = 9;

    ResidualPattern(
        const static_vector<Vec2, max_size> &newPattern = default_pattern)
        : _pattern(newPattern) {
      height =
          int(std::ceil(std::max_element(_pattern.begin(), _pattern.end(),
                                         [](const Vec2 &a, const Vec2 &b) {
                                           return a.lpNorm<Eigen::Infinity>() <
                                                  b.lpNorm<Eigen::Infinity>();
                                         })
                            ->lpNorm<Eigen::Infinity>()));
    }

    inline const static_vector<Vec2, max_size> &pattern() const {
      return _pattern;
    }
    int height;

  private:
    static_vector<Vec2, max_size> _pattern;
    static const static_vector<Vec2, max_size> default_pattern;
  } residualPattern;

  struct Intensity {
    static constexpr double default_outlierDiff = 12.0;
    double outlierDiff = default_outlierDiff;
  } intensity;

  struct Threading {
    static constexpr int default_numThreads = 4;
    int numThreads = default_numThreads;
  } threading;

  struct Optimization {
    struct StepControl {
      static constexpr double default_initialLambda = 1e-4;
      double initialLambda = default_initialLambda;

      static constexpr double default_initialFailMultiplier = 2;
      double initialFailMultiplier = default_initialFailMultiplier;

      static constexpr double default_failMultiplierMultiplier = 2;
      double failMultiplierMultiplier = default_failMultiplierMultiplier;

      static constexpr double default_minLambdaMultiplier = 1.0 / 3.0;
      double minLambdaMultiplier = default_minLambdaMultiplier;

      static constexpr double default_acceptedQuality = 0;
      double acceptedQuality = default_acceptedQuality;
    } stepControl;

    enum Loss { TRIVIAL, HUBER };

    static constexpr Loss default_lossType = HUBER;
    Loss lossType = default_lossType;

    static constexpr int default_maxIterations = 10;
    int maxIterations = default_maxIterations;

    static constexpr int default_maxConsecutiveFailedIterations = 10;
    int maxConsecutiveFailedIterations = default_maxConsecutiveFailedIterations;

    static constexpr bool default_fixedMotionOnFirstAdjustent = false;
    bool fixedMotionOnFirstAdjustent = default_fixedMotionOnFirstAdjustent;

    static constexpr bool default_fixedRotationOnSecondKF = false;
    bool fixedRotationOnSecondKF = default_fixedRotationOnSecondKF;

    static constexpr double default_minFirstToSecondRadius = 1e-2; // TODO
    double minFirstToSecondRadius = default_minFirstToSecondRadius;

    static constexpr double default_pointPointThres = 1e-7; // TODO
    double pointPointThres = default_pointPointThres;

    enum OptimizationType { DISABLED, SELF_WRITTEN, CERES };
    static constexpr OptimizationType default_optimizationType = SELF_WRITTEN;
    OptimizationType optimizationType = default_optimizationType;
  } optimization;

  static constexpr int default_maxOptimizedPoints = 2000;
  static constexpr int max_maxOptimizedPoints = 10000;
  inline int maxOptimizedPoints() const { return mMaxOptimizedPoints; }
  inline void setMaxOptimizedPoints(int newMaxOptimizedPoints) {
    CHECK(newMaxOptimizedPoints > 0 &&
          newMaxOptimizedPoints <= max_maxOptimizedPoints);
    mMaxOptimizedPoints = newMaxOptimizedPoints;
  }

  static constexpr double default_optimizedOnReprojectionFactor = 0.8;
  double optimizedOnReprojectionFactor = default_optimizedOnReprojectionFactor;

  static constexpr int default_maxKeyFrames = 7;
  inline int maxKeyFrames() const { return mMaxKeyFrames; }
  inline void setMaxKeyFrames(int newMaxKeyFrames) {
    CHECK(newMaxKeyFrames > 0);
    mMaxKeyFrames = newMaxKeyFrames;
  }
  int mMaxKeyFrames = default_maxKeyFrames;

  static constexpr int default_keyFrameDist = 4;
  static constexpr int max_keyFrameDist = 10;
  inline int keyFrameDist() const { return mKeyFrameDist; }
  inline void setKeyFrameDist(int newKeyFrameDist) {
    CHECK(newKeyFrameDist > 0 && newKeyFrameDist <= max_keyFrameDist);
    mKeyFrameDist = newKeyFrameDist;
  }

  static constexpr bool default_trackFromLastKf = true;
  bool trackFromLastKf = default_trackFromLastKf;

  static constexpr bool default_predictUsingScrew = true;
  bool predictUsingScrew = default_predictUsingScrew;

  static constexpr bool default_continueChoosingKeyFrames = true;
  bool continueChoosingKeyFrames = default_continueChoosingKeyFrames;

  static constexpr int default_expectedFramesCount = 3000;
  int expectedFramesCount = default_expectedFramesCount;

  static constexpr bool default_useRandomOptimizedChoice = false;
  int useRandomOptimizedChoice = default_useRandomOptimizedChoice;

  static constexpr bool default_disableMarginalization = false;
  int disableMarginalization = default_disableMarginalization;

  InitializerDelaunaySettings getInitializerDelaunaySettings() const;
  InitializerGroundTruthSettings getInitializerGroundTruthSettings() const;
  PointTracerSettings getPointTracerSettings() const;
  FrameTrackerSettings getFrameTrackerSettings() const;
  BundleAdjusterSettings getBundleAdjusterSettings() const;
  ResidualSettings getResidualSettings() const;
  EnergyFunctionSettings getEnergyFunctionSettings() const;

  Settings getDsoLikeSettings() const;
  Settings getGradientAdjustedSettings(double intencityRequiredToThis,
                                       double gradNormRequiredToThis) const;
  Settings getScaleAdjustedSettings(double scaleFactor) const;

private:
  int mKeyFrameDist = default_keyFrameDist;
  int mMaxOptimizedPoints = default_maxOptimizedPoints;
};

struct PointTracerSettings {
  Settings::PointTracer pointTracer = {};
  Settings::Intensity intensity = {};
  Settings::ResidualPattern residualPattern = {};
  Settings::Pyramid pyramid = {};
};

struct InitializerDelaunaySettings {
  Settings::DsoInitializer intializer = {};
  Settings::DelaunayDsoInitializer delaunayInitializer = {};
  Settings::StereoMatcher stereoMatcher = {};
  Settings::Threading threading = {};
  Settings::Triangulation triangulation = {};
  Settings::KeyFrame keyFrame = {};
};

struct InitializerGroundTruthSettings {
  Settings::DsoInitializer intializer;
  Settings::KeyFrame keyFrame;
  Settings::PixelSelector pixelSelector;
};

struct FrameTrackerSettings {
  Settings::FrameTracker frameTracker = {};
  Settings::Pyramid pyramid = {};
  Settings::AffineLight affineLight = {};
  Settings::Intensity intensity = {};
  Settings::ResidualWeighting residualWeighting = {};
  Settings::Threading threading = {};
};

struct ResidualSettings {
  inline int patternSize() const { return residualPattern.pattern().size(); }

  Settings::ResidualPattern residualPattern = {};
  Settings::ResidualWeighting residualWeighting = {};
  Settings::Intensity intensity = {};
  Settings::Depth depth = {};
};

struct EnergyFunctionSettings {
  ResidualSettings residual = {};
  Settings::AffineLight affineLight = {};
  Settings::Optimization optimization = {};
  Settings::Depth depth = {};
};

struct BundleAdjusterSettings {
  Settings::Optimization optimization = {};
  Settings::ResidualPattern residualPattern = {};
  Settings::ResidualWeighting residualWeighting = {};
  Settings::Intensity intensity = {};
  Settings::AffineLight affineLight = {};
  Settings::Threading threading = {};
  Settings::Depth depth = {};
  EnergyFunctionSettings energyFunction = {};
};

} // namespace mdso

#endif
