#ifndef INCLUDE_INITIALIZEROBSERVER
#define INCLUDE_INITIALIZEROBSERVER

#include "system/KeyFrame.h"
#include "util/SphericalTerrain.h"
#include <vector>

namespace fishdso {

class InitializerObserver {
public:
  virtual void initialized(const KeyFrame *lastKeyFrame,
                           const SphericalTerrain *lastTerrain,
                           const StdVector<Vec2> &keyPoints,
                           const std::vector<double> &keyPointDepths) {}

protected:
  InitializerObserver() {}
};

} // namespace fishdso

#endif
