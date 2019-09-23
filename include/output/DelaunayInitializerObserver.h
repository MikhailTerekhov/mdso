#ifndef INCLUDE_INITIALIZEROBSERVER
#define INCLUDE_INITIALIZEROBSERVER

#include "system/KeyFrame.h"
#include "util/SphericalTerrain.h"
#include <vector>

namespace mdso {

struct InitializedFrame;

class DelaunayInitializerObserver {
public:
  virtual ~DelaunayInitializerObserver() = 0;

  virtual void initialized(const InitializedFrame frames[],
                           const SphericalTerrain terrains[], Vec2 *keyPoints[],
                           double *keyPointDepths[], int sizes[]) {}
};

} // namespace mdso

#endif
