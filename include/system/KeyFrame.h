#ifndef INCLUDE_KEYFRAME
#define INCLUDE_KEYFRAME

#include "system/DsoInitializer.h"
#include "system/ImmaturePoint.h"
#include "system/OptimizedPoint.h"
#include "system/PreKeyFrame.h"
#include "util/DepthedImagePyramid.h"
#include "util/PixelSelector.h"
#include "util/settings.h"
#include <Eigen/StdVector>
#include <memory>
#include <opencv2/core.hpp>
#include <vector>

namespace mdso {

struct KeyFrame;

struct KeyFrameEntry {
  KeyFrameEntry(const InitializedFrame::FrameEntry &entry, KeyFrame *host,
                int ind, const PointTracerSettings &tracingSettings);
  KeyFrameEntry(KeyFrame *host, int ind, Timestamp timestamp);

  StdVector<ImmaturePoint> immaturePoints;
  StdVector<OptimizedPoint> optimizedPoints;

  KeyFrame *host;
  int ind;
  Timestamp timestamp;
  AffLight lightWorldToThis;
};

struct KeyFrame {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  KeyFrame(const KeyFrame &other) = delete;
  KeyFrame(KeyFrame &&other) = delete;
  KeyFrame(const InitializedFrame &initializedFrame, CameraBundle *cam,
           Preprocessor *preprocessor, int globalFrameNum,
           PixelSelector pixelSelector[],
           const Settings::KeyFrame &_kfSettings = {},
           const Settings::Pyramid &pyrSettings = {},
           const PointTracerSettings &tracingSettings = {});
  KeyFrame(std::unique_ptr<PreKeyFrame> newPreKeyFrame,
           PixelSelector pixelSelector[],
           const Settings::KeyFrame &_kfSettings = {},
           const PointTracerSettings &tracingSettings = {});

  std::unique_ptr<PreKeyFrame> preKeyFrame;
  SE3 thisToWorld;
  std::vector<KeyFrameEntry> frames;
  std::vector<std::unique_ptr<PreKeyFrame>> trackedFrames;

  Settings::KeyFrame kfSettings;

private:
  void addImmatures(const cv::Point points[], int size, int numInBundle,
                    const PointTracerSettings &tracingSettings);
};

} // namespace mdso

#endif
