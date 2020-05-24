#ifndef INCLUDE_GETMFOVCAM
#define INCLUDE_GETMFOVCAM

#include "system/CameraModel.h"
#include "util/types.h"

namespace mdso {

CameraModel getMfovCam(const fs::path &intrinsicsFname);
}

#endif
