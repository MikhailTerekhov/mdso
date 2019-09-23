#include "util/Sim3Aligner.h"
#include <glog/logging.h>

namespace mdso {

Sim3Aligner::Sim3Aligner(const SE3 &worldToFirst, const SE3 &worldToSecond,
                         const SE3 &worldToFirstGT,
                         const SE3 &worldToSecondGT) {
  SE3 fToL = worldToSecond * worldToFirst.inverse();
  SE3 fToLGT = worldToSecondGT * worldToFirstGT.inverse();
  if (fToLGT.translation().norm() < 1e-3 || fToL.translation().norm() < 1e-3) {
    LOG(WARNING) << "Could not align GT poses as motion is too static"
                 << std::endl;
    return;
  }

  scaleGTToDso = fToL.translation().norm() / fToLGT.translation().norm();
  SE3 worldToFirstGTScaled = worldToFirstGT;
  worldToFirstGTScaled.translation() *= scaleGTToDso;
  gtToDso = worldToFirstGTScaled.inverse() * worldToFirst;
}

SE3 Sim3Aligner::alignWorldToFrameGT(const SE3 &worldToFrameGT) const {
  SE3 result = worldToFrameGT;
  result.translation() *= scaleGTToDso;
  result = result * gtToDso;
  return result;
}

Vec3 Sim3Aligner::alignScale(const Vec3 &pointInFrameGT) const {
  return scaleGTToDso * pointInFrameGT;
}

} // namespace mdso
