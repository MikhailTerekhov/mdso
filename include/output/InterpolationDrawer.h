#ifndef INCLUDE_INTERPOLATIONDRAWER
#define INCLUDE_INTERPOLATIONDRAWER

#include "output/InitializerObserver.h"

namespace fishdso {

class InterpolationDrawer : public InitializerObserver {
public:
  InterpolationDrawer(CameraModel *cam);

  void initialized(const KeyFrame *lastKeyFrame,
                   const SphericalTerrain *lastTerrain,
                   const StdVector<Vec2> &keyPoints,
                   const std::vector<double> &keyPointDepths);

  bool didInitialize();
  cv::Mat3b draw();

private:
  CameraModel *cam;
  bool mDidInitialize;
  cv::Mat3b result;
};

} // namespace fishdso

#endif
