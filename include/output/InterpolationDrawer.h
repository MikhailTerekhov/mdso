#ifndef INCLUDE_INTERPOLATIONDRAWER
#define INCLUDE_INTERPOLATIONDRAWER

#include "output/DelaunayInitializerObserver.h"

namespace mdso {

class InterpolationDrawer : public DelaunayInitializerObserver {
public:
  InterpolationDrawer(CameraModel *cam);

  void initialized(const InitializedFrame frames[],
                   const SphericalTerrain terrains[], Vec2 *keyPoints[],
                   double *keyPointDepths[], int sizes[]) override;

  bool didInitialize();
  cv::Mat3b draw();

private:
  CameraModel *cam;
  bool mDidInitialize;
  cv::Mat3b result;
};

} // namespace mdso

#endif
