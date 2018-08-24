#pragma once

namespace fishdso {

#define settingEssentialMinimalSolveN 5

// for candidate point selection
extern double settingGradThreshold1;
extern double settingGradThreshold2;
extern double settingGradThreshold3;

extern int settingInitialAdaptiveBlockSize;
extern int settingInterestPointsAdaptTo;
extern int settingInterestPointsUsed;

// parameters of the catadioptric camera projection algorithm
extern int settingCameraMapPolyDegree;
extern int settingCameraMapPolyPoints;

// parameters of the orb-keypoints-based initialization
extern double settingInitKeypointsObserveAngle;
extern float settingMatchNonMove;
extern int settingFirstFramesSkip;
extern double settingOrbInlierProb;
extern double settingEssentialSuccessProb;
extern double settingEssentialReprojErrThreshold;
extern double settingRemoveResidualsRatio;

// used for undistortion
extern int settingHalfFillingFilterSize;

} // namespace fishdso
