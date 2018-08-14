#pragma once

#include "../system/cameramodel.h"
#include <opencv2/core.hpp>

namespace fishdso {

// Algorithm that is used to project points in catadioptric camera model
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

extern int settingCameraMapPolyDegree;
extern int settingCameraMapPolyPoints;

} // namespace fishdso
