#ifndef INCLUDE_OBSERVERS
#define INCLUDE_OBSERVERS

#include <vector>

namespace fishdso {

class DsoObserver;
class FrameTrackerObserver;
class InitializerObserver;

struct Observers {
  std::vector<DsoObserver *> dso;
  std::vector<InitializerObserver *> initializer;
  std::vector<FrameTrackerObserver *> frameTracker;
};

} // namespace fishdso

#endif
