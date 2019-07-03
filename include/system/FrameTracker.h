#ifndef INCLUDE_FRAMETRACKER
#define INCLUDE_FRAMETRACKER

#include "system/AffineLightTransform.h"
#include "system/CameraModel.h"
#include "system/KeyFrame.h"
#include "util/DepthedImagePyramid.h"

namespace fishdso {

class FrameTrackerObserver;

class FrameTracker {
public:
  FrameTracker(const StdVector<CameraModel> &camPyr,
               std::unique_ptr<DepthedImagePyramid> _baseFrame,
               const std::vector<FrameTrackerObserver *> &observers = {},
               const FrameTrackerSettings &_settings = {});

  std::pair<SE3, AffineLightTransform<double>>
  trackFrame(const ImagePyramid &frame, const SE3 &coarseMotion,
             const AffineLightTransform<double> &coarseAffLight);

  void addObserver(FrameTrackerObserver *observer);

  // output only
  std::vector<cv::Mat3b> residualsImg;

  double lastRmse;

private:
  std::pair<SE3, AffineLightTransform<double>>
  trackPyrLevel(const CameraModel &cam, const cv::Mat1b &baseImg,
                const cv::Mat1d &baseDepths, const cv::Mat1b &trackedImg,
                const SE3 &coarseMotion,
                const AffineLightTransform<double> &coarseAffLight,
                int pyrLevel);

  const StdVector<CameraModel> &camPyr;
  std::unique_ptr<DepthedImagePyramid> baseFrame;
  int displayWidth, displayHeight;

  std::vector<FrameTrackerObserver *> observers;
  FrameTrackerSettings settings;
};

struct PointTrackingResidual {
  PointTrackingResidual(
      Vec3 pos, double baseIntensity, const CameraModel *cam,
      ceres::BiCubicInterpolator<ceres::Grid2D<unsigned char, 1>> *trackedFrame)
      : pos(pos)
      , baseIntensity(baseIntensity)
      , cam(cam)
      , trackedFrame(trackedFrame) {}

  template <typename T>
  bool operator()(const T *const rotP, const T *const transP,
                  const T *const affLightP, T *res) const {
    typedef Eigen::Matrix<T, 2, 1> Vec2t;
    typedef Eigen::Matrix<T, 3, 1> Vec3t;
    typedef Eigen::Matrix<T, 3, 3> Mat33t;
    typedef Eigen::Quaternion<T> Quatt;
    typedef Sophus::SE3<T> SE3t;

    Eigen::Map<const Vec3t> transM(transP);
    Vec3t trans(transM);
    Eigen::Map<const Quatt> rotM(rotP);
    Quatt rot(rotM);
    SE3t motion(rot, trans);
    AffineLightTransform<T> affLight(affLightP[0], affLightP[1]);

    Vec3t newPos = motion * pos.cast<T>();
    Vec2t newPosProj = cam->map(newPos.data());

    T trackedIntensity;
    trackedFrame->Evaluate(newPosProj[1], newPosProj[0], &trackedIntensity);
    res[0] = affLight(trackedIntensity) - baseIntensity;

    return true;
  }

  Vec3 pos;
  double baseIntensity;
  const CameraModel *cam;
  ceres::BiCubicInterpolator<ceres::Grid2D<unsigned char, 1>> *trackedFrame;
};

} // namespace fishdso

#endif
