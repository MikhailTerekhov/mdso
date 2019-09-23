#ifndef INCLUDE_OBSERVERS
#define INCLUDE_OBSERVERS

#include <vector>

namespace mdso {

class DsoObserver;
class FrameTrackerObserver;
class DelaunayInitializerObserver;

struct Observers {
  std::vector<DsoObserver *> dso;
  std::vector<DelaunayInitializerObserver *> initializer;
  std::vector<FrameTrackerObserver *> frameTracker;
};

} // namespace mdso

#endif
