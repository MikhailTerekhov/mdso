#ifndef INCLUDE_INITIALIZEROBSERVER
#define INCLUDE_INITIALIZEROBSERVER

#include "system/KeyFrame.h"
#include "util/SphericalTerrain.h"
#include <vector>

namespace fishdso {

struct InitializedFrame;

class DelaunayInitializerObserver {
public:
  virtual ~DelaunayInitializerObserver() = 0;

  virtual void initialized(const InitializedFrame *lastFrame,
                           const SphericalTerrain *lastTerrain,
                           Vec2 keyPoints[], double keyPointDepths[],
                           int size) {}
};

} // namespace fishdso

#endif
