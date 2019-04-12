#ifndef INCLUDE_DEBUGIMAGEDRAWER
#define INCLUDE_DEBUGIMAGEDRAWER

#include "output/DsoObserver.h"
#include "output/TrackingDebugImageDrawer.h"
#include "system/DsoSystem.h"

DECLARE_double(debug_rel_point_size);
DECLARE_int32(debug_image_width);
DECLARE_double(debug_max_stddev);

namespace fishdso {

class DebugImageDrawer : public DsoObserver {
public:
  DebugImageDrawer();

  void created(DsoSystem *newDso, CameraModel *newCam,
               const Settings &newSettings);
  void newFrame(const PreKeyFrame *newFrame);
  void newKeyFrame(const KeyFrame *newBaseFrame);

  cv::Mat3b draw();

private:
  DsoSystem *dso;
  CameraModel *cam;
  Settings settings;
  const KeyFrame *baseFrame;
  const PreKeyFrame *lastFrame;
  std::unique_ptr<TrackingDebugImageDrawer> residualsDrawer;
};

} // namespace fishdso

#endif
