#ifndef INCLUDE_CAMERABUNDLE
#define INCLUDE_CAMERABUNDLE

#include "system/CameraModel.h"

namespace mdso {

struct CameraBundle {
  using CamPyr = static_vector<CameraBundle, Settings::Pyramid::max_levelNum>;

  struct CameraEntry {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    CameraEntry(const SE3 &_bodyToThis, const CameraModel &cam);

    SE3 bodyToThis;
    SE3 thisToBody;
    CameraModel cam;
  };

  CameraBundle(SE3 bodyToCam[], CameraModel cam[], int size);

  void setCamToBody(int ind, const SE3 &camToBody);
  CamPyr camPyr(int levelNum) const;

  StdVector<CameraEntry> bundle;
};

} // namespace mdso

#endif
