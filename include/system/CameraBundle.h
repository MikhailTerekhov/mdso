#ifndef INCLUDE_CAMERABUNDLE
#define INCLUDE_CAMERABUNDLE

#include "system/CameraModel.h"

namespace fishdso {

struct CameraBundle {
  using CamPyr = static_vector<CameraBundle, Settings::Pyramid::max_levelNum>;

  struct CameraEntry {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    CameraEntry(const SE3 &_bodyToThis, const CameraModel &cam);

    const SE3 bodyToThis;
    const SE3 thisToBody;
    CameraModel cam;
  };

  CameraBundle(SE3 bodyToCam[], CameraModel cam[], int size);

  CamPyr camPyr(int levelNum);

  StdVector<CameraEntry> bundle;
};

} // namespace fishdso

#endif
