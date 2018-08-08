#pragma once

#include <opencv2/core.hpp>

namespace fishdso {

// for candidate point selection
extern double settingGradThreshold1;
extern double settingGradThreshold2;
extern double settingGradThreshold3;

extern int settingInitialAdaptiveBlockSize;
extern int settingInterestPointsAdaptTo;
extern int settingInterestPointsUsed;

} // namespace fishdso
