#include "system/CameraBundle.h"

namespace fishdso {

CameraBundle::CameraEntry::CameraEntry(const SE3 &_bodyToThis, const CameraModel &cam)
    : bodyToThis(_bodyToThis)
    , thisToBody(_bodyToThis.inverse())
    , cam(cam) {}

CameraBundle::CameraBundle(SE3 bodyToCam[], CameraModel cam[], int size) {
  for (int i = 0; i < size; ++i)
    bundle.emplace_back(bodyToCam[i], cam[i]);
}

CameraBundle::CamPyr CameraBundle::camPyr(int pyrLevels) {
  CHECK(pyrLevels > 0 && pyrLevels <= Settings::Pyramid::max_levelNum);

  CameraModel::CamPyr pyramids[Settings::CameraBundle::max_camerasInBundle];
  for (int ci = 0; ci < bundle.size(); ++ci)
    pyramids[ci] = bundle[ci].cam.camPyr(pyrLevels);

  CameraBundle::CamPyr result(pyrLevels);
  for (int pl = 0; pl < pyrLevels; ++pl)
    for (int ci = 0; ci < bundle.size(); ++ci)
      result[pl].bundle.push_back({bundle[ci].bodyToThis, pyramids[ci][pl]});

  return result;
}

} // namespace fishdso
