#ifndef INCLUDE_DEBUGIMAGEDRAWER
#define INCLUDE_DEBUGIMAGEDRAWER

#include "output/DsoObserver.h"
#include "output/TrackingDebugImageDrawer.h"
#include "system/DsoSystem.h"
#include "system/Reprojector.h"

DECLARE_double(debug_rel_point_size);
DECLARE_int32(debug_image_width);
DECLARE_double(debug_max_stddev);

namespace mdso {

class DebugImageDrawer : public DsoObserver {
public:
  class DrawingSettings {
  public:
    DrawingSettings(int numCameras);

    const std::vector<int> &drawingOrder() const { return mDrawingOrder; }
    void setDrawingOrder(const std::vector<int> &newDrawingOrder);

    bool useOldLayoutForSingleCamera = true;

  private:
    std::vector<int> mDrawingOrder;
  };

  DebugImageDrawer(const std::vector<int> &drawingOrder);
  DebugImageDrawer(const DrawingSettings &drawingSettings);

  void created(DsoSystem *newDso, CameraBundle *newCam,
               const Settings &newSettings) override;
  void newFrame(const PreKeyFrame &frame) override;
  void newBaseFrame(const KeyFrame &newBaseFrame) override;

  bool isDrawable() const;
  cv::Mat3b draw();

private:
  std::vector<cv::Mat3b>
  drawProjDepths(const StdVector<Reprojection> &immatures,
                 const StdVector<Reprojection> &optimized) const;
  std::vector<cv::Mat3b>
  drawUseful(const std::vector<const KeyFrame *> &keyFrames,
             const StdVector<Reprojection> &optimized) const;
  std::vector<cv::Mat3b>
  drawStddevs(const std::vector<const KeyFrame *> &keyFrames,
              const StdVector<Reprojection> &immatures,
              const StdVector<Reprojection> &optimized) const;

  DsoSystem *dso = nullptr;
  CameraBundle *cam = nullptr;
  CameraBundle::CamPyr camPyr;
  Settings settings;
  const KeyFrame *baseFrame = nullptr;
  const PreKeyFrame *lastFrame = nullptr;
  std::unique_ptr<TrackingDebugImageDrawer> residualsDrawer;
  DrawingSettings drawingSettings;
};

} // namespace mdso

#endif
