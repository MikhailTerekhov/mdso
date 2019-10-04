#ifndef INCLUDE_FRAMETRACKER
#define INCLUDE_FRAMETRACKER

#include "system/AffineLightTransform.h"
#include "system/CameraModel.h"
#include "system/KeyFrame.h"
#include "util/DepthedImagePyramid.h"

namespace mdso {

class FrameTrackerObserver;

struct TrackingResult {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  TrackingResult(int camNumber);

  SE3 baseToTracked;
  std::vector<AffLight> lightBaseToTracked;
};

class FrameTracker {
public:
  using DepthedMultiFrame = std::vector<DepthedImagePyramid>;

  FrameTracker(CameraBundle camPyr[], const DepthedMultiFrame &baseFrame,
               const KeyFrame &baseFrameAsKf,
               std::vector<FrameTrackerObserver *> &observers,
               const FrameTrackerSettings &_settings = {});

  TrackingResult trackFrame(const PreKeyFrame &frame,
                            const TrackingResult &coarseTrackingResult);

private:
  class DepthPyramidSlice {
  public:
    struct Entry {
      struct Point {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        Point(const DepthedImagePyramid::Point &p, const CameraModel &cam,
              const cv::Mat1b &img);

        Vec2 p;
        double depth;
        Vec3 ray;
        double gradNorm;
        double intensity;
      };

      Entry(const DepthedMultiFrame &frame, const CameraModel &cam,
            int levelNum, int cameraNum);

      StdVector<Point> points;
    };

    DepthPyramidSlice(const DepthedMultiFrame &frame, const CameraBundle &cam,
                      int levelNum);

    Entry &operator[](int ind);
    const Entry &operator[](int ind) const;
    int totalPoints() const;

  private:
    std::vector<Entry> entries;
    int mTotalPoints;
  };

  TrackingResult trackPyrLevel(const PreKeyFrame &frame,
                               const TrackingResult &coarseTrackingResult,
                               int pyrLevel);

  CameraBundle *camPyr;
  std::vector<DepthPyramidSlice> baseFrameSlices;
  std::vector<FrameTrackerObserver *> &observers;
  std::vector<std::vector<AffLight>> baseAffLightFromTo;
  FrameTrackerSettings settings;
};

} // namespace mdso

#endif
