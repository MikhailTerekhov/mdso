#ifndef INCLUDE_DISTANCEMAP
#define INCLUDE_DISTANCEMAP

#include "system/CameraBundle.h"
#include "util/settings.h"
#include "util/types.h"

namespace mdso {

class DistanceMap {
public:
  DistanceMap(CameraBundle *cam, StdVector<Vec2> points[],
              const Settings::DistanceMap &settings = {});

  int choose(StdVector<Vec2> otherPoints[], int pointsNeeded,
             std::vector<int> chosenIndices[]);

private:
  struct MapEntry {
    MapEntry(int givenW, int givenH, const StdVector<Vec2> &points,
             const Settings::DistanceMap &settings);
    int pyrDown;
    MatXXi dist;
  };

  int camCount;
  std::vector<MapEntry> maps;

  Settings::DistanceMap settings;
};

} // namespace mdso

#endif
