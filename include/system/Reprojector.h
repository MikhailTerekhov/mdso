#ifndef INCLUDE_REPROJECTOR
#define INCLUDE_REPROJECTOR

#include "system/KeyFrame.h"

namespace mdso {

struct Reprojection {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  int hostInd;
  int hostCamInd;
  int targetCamInd;
  int pointInd;
  Vec2 reprojected;
  double reprojectedDepth;
};

struct DepthedPoints {
  DepthedPoints(int numCams, int totalReprojected);

  std::vector<StdVector<Vec2>> points;
  std::vector<std::vector<double>> depths;
  std::vector<std::vector<double>> weights;
};

template <typename PointType> class Reprojector {
public:
  Reprojector(const KeyFrame *const *keyFrames, int numKeyFrames,
              const SE3 &targetBodyToWorld,
              const Settings::Depth &depthSettings, int borderSize = 0)
      : targetWorldToBody(targetBodyToWorld.inverse())
      , keyFrames(keyFrames, keyFrames + numKeyFrames)
      , cam(numKeyFrames <= 0 ? nullptr : keyFrames[0]->preKeyFrame->cam)
      , borderSize(borderSize)
      , numCams(cam->bundle.size())
      , numKeyFrames(numKeyFrames)
      , depthSettings(depthSettings) {
    CHECK_GE(numKeyFrames, 0);
  }

  StdVector<Reprojection> reproject() const;
  DepthedPoints reprojectDepthed() const;

private:
  static const StdVector<PointType> &getPoints(const KeyFrameEntry &entry);
  static double getDepth(const PointType &p);

  SE3 targetWorldToBody;
  std::vector<const KeyFrame *> keyFrames;
  CameraBundle *cam;
  int borderSize;
  int numCams;
  int numKeyFrames;
  Settings::Depth depthSettings;
};

} // namespace mdso

#endif
