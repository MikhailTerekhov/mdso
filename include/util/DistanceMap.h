#ifndef INCLUDE_DISTANCEMAP
#define INCLUDE_DISTANCEMAP

#include "system/CameraBundle.h"
#include "util/settings.h"
#include "util/types.h"

namespace fishdso {

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
  static_vector<MapEntry, Settings::CameraBundle::max_camerasInBundle> maps;

  Settings::DistanceMap settings;
};

} // namespace fishdso

#endif
