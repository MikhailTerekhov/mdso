#include "settings.h"

#include <opencv2/core.hpp>

namespace fishdso {

// Point-of-interest detector
double settingGradThreshold1 = 30.0;
double settingGradThreshold2 = 6.0;
double settingGradThreshold3 = 3.0;

int settingInitialAdaptiveBlockSize = 25;
int settingInterestPointsAdaptTo = 2150;
int settingInterestPointsUsed = 2000;

int settingCameraMapPolyDegree = 7;
int settingCameraMapPolyPoints = 2000;

} // namespace fishdso
