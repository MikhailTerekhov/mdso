#include "internal/data/getMfovCam.h"

namespace mdso {

CameraModel getMfovCam(const fs::path &intrinsicsFname) {
  // Our CameraModel is partially compatible with the provided one (affine
  // transformation used in omni_cam is just scaling in our case, but no problem
  // raises since in this dataset no affine transformation is happening). We
  // also compute the inverse polynomial ourselves instead of using the provided
  // one.

  int width, height;
  double unmapPolyCoeffs[5];
  Vec2 center;
  std::ifstream camIfs(intrinsicsFname);
  CHECK(camIfs.is_open()) << "could not open camera intrinsics file \""
                          << intrinsicsFname.native() << "\"";
  camIfs >> width >> height;
  for (int i = 0; i < 5; ++i)
    camIfs >> unmapPolyCoeffs[i];
  VecX ourCoeffs(4);
  ourCoeffs << unmapPolyCoeffs[0], unmapPolyCoeffs[2], unmapPolyCoeffs[3],
      unmapPolyCoeffs[4];
  ourCoeffs *= -1;
  camIfs >> center[0] >> center[1];
  return CameraModel(width, height, 1.0, center, ourCoeffs);
}

} // namespace mdso
