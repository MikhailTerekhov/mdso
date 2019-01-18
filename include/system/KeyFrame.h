#ifndef INCLUDE_KEYFRAME
#define INCLUDE_KEYFRAME

#include "system/ImmaturePoint.h"
#include "system/OptimizedPoint.h"
#include "system/PreKeyFrame.h"
#include "util/DepthedImagePyramid.h"
#include "util/settings.h"
#include <Eigen/StdVector>
#include <memory>
#include <opencv2/core.hpp>
#include <vector>

namespace fishdso {

struct KeyFrame {
  static constexpr int LI = settingInterestPointLayers;
  static constexpr int PL = settingPyrLevels;
  static int adaptiveBlockSize;

  KeyFrame(const KeyFrame &other) = delete;
  KeyFrame(KeyFrame &&other) = default;
  KeyFrame(CameraModel *cam, const cv::Mat &frameColored, int globalFrameNum);
  KeyFrame(std::shared_ptr<PreKeyFrame> newPreKeyFrame);

  void activateAllImmature();
  void deactivateAllOptimized();

  std::unique_ptr<DepthedImagePyramid> makePyramid();
  void selectPointsDenser(int pointsNeeded);

  cv::Mat drawDepthedFrame(double minDepth, double maxDepth);

  std::shared_ptr<PreKeyFrame> preKeyFrame;

  StdUnorderedSet<std::unique_ptr<ImmaturePoint>> immaturePoints;
  StdUnorderedSet<std::unique_ptr<OptimizedPoint>> optimizedPoints;

private:
  static void updateAdaptiveBlockSize(int pointsFound);

  int selectPoints(int blockSize, int pointsNeeded);

  int lastBlockSize;
  int lastPointsFound;
  int lastPointsUsed;
};

} // namespace fishdso

#endif
