#include "optimize/parametrizations.h"
#include <glog/logging.h>

namespace mdso::optimize {

S2Parametrization::S2Parametrization(const Vec3t &center,
                                     const Vec3t &initialValue)
    : mValue(initialValue)
    , mCenter(center)
    , mRadius((initialValue - center).norm()) {
  CHECK_GE(mRadius, 1e-4) << "Zero radius for S2 parametrization";
  recalcOrts();
}

S2Parametrization::MatDiff S2Parametrization::diffPlus() const {
  return 0.5 * localToWorldRot.leftCols<2>();
}

void S2Parametrization::addDelta(const Tangent &delta) {
  using boost::math::sinc_pi;
  Vec3t shift;
  Vec2t deltaN = delta / mRadius;
  const T deltaNormBy2r = deltaN.norm() / 2;
  shift.head<2>() = deltaN * (0.5 * sinc_pi(deltaNormBy2r));
  shift[2] = cos(deltaNormBy2r);
  Vec3t shiftRot = localToWorldRot * shift;
  mValue = shiftRot * mRadius + mCenter;
  recalcOrts();
}

void S2Parametrization::recalcOrts() {
  Vec3t v = (mValue - mCenter).normalized();
  int minI = std::min_element(
                 v.data(), v.data() + 3,
                 [](double a, double b) { return std::abs(a) < std::abs(b); }) -
             v.data();
  Vec3t v1, v2;
  if (minI == 0)
    v1 = Vec3(0, -v[2], v[1]).normalized();
  else if (minI == 1)
    v1 = Vec3(-v[2], 0, v[0]).normalized();
  else
    v1 = Vec3(-v[1], v[0], 0).normalized();
  v2 = v.cross(v1).normalized();
  localToWorldRot << v1, v2, v;
}

SO3xS2Parametrization::SO3xS2Parametrization(const SE3t &f1ToWorld,
                                             const SE3t &f2ToWorld)
    : mSo3(f2ToWorld.so3())
    , mS2(f1ToWorld.translation(), f2ToWorld.translation()) {}

SO3xS2Parametrization::SO3xS2Parametrization(const SO3t &baseRot,
                                             const Vec3t &centerTrans,
                                             const Vec3t &initialTrans)
    : mSo3(baseRot)
    , mS2(centerTrans, initialTrans) {}

SO3xS2Parametrization::MatDiff SO3xS2Parametrization::diffPlus() const {
  MatDiff result = MatDiff::Zero();
  result.topLeftCorner<4, 3>() = mSo3.diffPlus();
  result.bottomRightCorner<3, 2>() = mS2.diffPlus();
  return result;
}

void SO3xS2Parametrization::addDelta(
    const SO3xS2Parametrization::Tangent &delta) {
  mSo3.addDelta(delta.head<3>());
  mS2.addDelta(delta.tail<2>());
}

} // namespace mdso::optimize
