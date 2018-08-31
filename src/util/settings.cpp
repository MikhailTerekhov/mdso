#include "util/settings.h"
#include "util/defs.h"

#include <opencv2/core.hpp>

namespace fishdso {

// Point-of-interest detector
double settingGradThreshold[settingInterestPointLayers] = {30.0, 6.0, 3.0};

int settingInitialAdaptiveBlockSize = 25;
int settingInterestPointsAdaptTo = 2150;
int settingInterestPointsUsed = 2000;

int settingCameraMapPolyDegree = 10;
int settingCameraMapPolyPoints = 2000;

int settingKeyPointsCount = 2000;
int settingRansacMaxIter = 100000;
double settingInitKeypointsObserveAngle = M_PI / 3;
float settingMatchNonMove = 9.0;
int settingFirstFramesSkip = 0;
double settingEssentialSuccessProb = 0.99;
double settingEssentialReprojErrThreshold = 4.0;
double settingRemoveResidualsRatio = 0.5;

int settingHalfFillingFilterSize = 1;

double settingEpsPointIsOnSegment = 1e-9;
double settingEpsSamePoints = 1e-9;

double settingTriangulationDrawPadding = 0.1;

} // namespace fishdso
