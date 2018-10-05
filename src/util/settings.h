#pragma once

namespace fishdso {

// candidate point selection
constexpr int settingInterestPointLayers = 3;
extern double settingGradThreshold[settingInterestPointLayers];
extern int settingInitialAdaptiveBlockSize;
extern int settingInterestPointsAdaptTo;
extern int settingInterestPointsUsed;

// catadioptric camera projection algorithm
extern int settingCameraMapPolyDegree;
extern int settingCameraMapPolyPoints;

// orb-keypoints-based initialization
extern int settingKeyPointsCount;
extern int settingRansacMaxIter;
extern double settingInitKeypointsObserveAngle;
extern double settingMatchNonMove;
extern int settingFirstFramesSkip;
constexpr int settingEssentialMinimalSolveN = 5;
extern double settingEssentialSuccessProb;
extern double settingEssentialReprojErrThreshold;
extern double settingRemoveResidualsRatio;

// undistortion
extern int settingHalfFillingFilterSize;

// triangulation
extern double settingEpsPointIsOnSegment;
extern double settingEpsSamePoints;

extern double settingTriangulationDrawPadding;

// frame tracking
constexpr int settingPyrLevels = 6;
extern double settingOutlierIntensityDiff;

} // namespace fishdso
