#ifndef INCLUDE_DISTANCEMAP
#define INCLUDE_DISTANCEMAP

#include "util/settings.h"
#include "util/types.h"

namespace fishdso {

class DistanceMap {
public:
  DistanceMap(int givenW, int givenH, const StdVector<Vec2> &points,
              const Settings::DistanceMap &settings = {});

  // returns coefficients chosen
  std::vector<int> choose(const StdVector<Vec2> &otherPoints, int pointsNeeded);

private:
  int pyrDown;
  MatXXi dist;

  Settings::DistanceMap settings;
};

} // namespace fishdso

#endif
