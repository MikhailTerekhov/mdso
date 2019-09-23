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

  void created(DsoSystem *newDso, CameraBundle *newCam,
               const Settings &newSettings) override;
  void newFrame(const PreKeyFrame &frame) override;
  void newBaseFrame(const KeyFrame &newBaseFrame) override;

  cv::Mat3b draw();

private:
  cv::Mat3b drawProjDepths(const StdVector<Vec2> &optProj,
                           const std::vector<double> &optDepths,
                           const StdVector<Vec2> &immProj,
                           const std::vector<ImmaturePoint *> &immRefs,
                           const std::vector<double> &immDepths);
  cv::Mat3b drawUseful(const StdVector<Vec2> &optBaseProj,
                       const std::vector<OptimizedPoint *> &optBaseRefs);
  cv::Mat3b drawStddevs(const StdVector<Vec2> &optProj,
                        const std::vector<OptimizedPoint *> &optRefs,
                        const StdVector<Vec2> &immProj,
                        const std::vector<ImmaturePoint *> &immRefs);

  DsoSystem *dso;
  CameraBundle *cam;
  CameraBundle::CamPyr camPyr;
  Settings settings;
  const KeyFrame *baseFrame;
  const PreKeyFrame *lastFrame;
  std::unique_ptr<TrackingDebugImageDrawer> residualsDrawer;
};

} // namespace fishdso

#endif
