#pragma once

namespace fishdso {

// algorithm that is used to project points in catadioptric camera model
#define CAMERA_MAP_POLYNOMIAL_Z 0
#define CAMERA_MAP_POLYNOMIAL_ANGLE 1
#define CAMERA_MAP_TYPE CAMERA_MAP_POLYNOMIAL_ANGLE

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
extern double settingFeatureMatchThreshold;
extern int settingFirstFramesSkip;
extern int settingEssentialMinimalSolveN;
extern double settingOrbInlierProb;
extern double settingEssentialSuccessProb;
extern double settingEssentialReprojErrThreshold;

// used for undistortion
extern int settingHalfFillingFilterSize;

} // namespace fishdso
