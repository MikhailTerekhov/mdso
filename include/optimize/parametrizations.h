#ifndef INCLUDE_PARAMETRIZATIONS
#define INCLUDE_PARAMETRIZATIONS

#include "util/types.h"
#include <boost/math/special_functions/sinc.hpp>

namespace mdso::optimize {

template <typename LieGroup> class RightExpParametrization {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  static constexpr int DoF = LieGroup::DoF;
  static constexpr int num_parameters = LieGroup::num_parameters;

  using MatDiff =
      Eigen::Matrix<typename LieGroup::Scalar, LieGroup::num_parameters, DoF>;
  using Tangent = typename LieGroup::Tangent;

  RightExpParametrization(const LieGroup &initialValue)
      : mValue(initialValue) {}

  MatDiff diffPlus() const { return mValue.Dx_this_mul_exp_x_at_0(); }
  void addDelta(const Tangent &delta) { mValue *= LieGroup::exp(delta); }
  inline LieGroup value() const { return mValue; }
  T *data() { return mValue.data(); }

private:
  LieGroup mValue;
};

class S2Parametrization {
public:
  static constexpr int DoF = 2;
  static constexpr int num_parameters = 3;
  using MatDiff = Eigen::Matrix<T, num_parameters, DoF>;
  using Tangent = Vec2t;

  S2Parametrization(const Vec3t &center, const Vec3t &initialValue);

  MatDiff diffPlus() const;
  void addDelta(const Tangent &delta);

  inline Vec3t value() const { return mValue; }
  inline Vec3t center() const { return mCenter; }
  inline T radius() const { return mRadius; }
  inline T *data() { return mValue.data(); }

private:
  void recalcOrts();

  Vec3t mValue;
  Vec3t mCenter;
  T mRadius;
  Mat33t localToWorldRot;
};

class SO3xS2Parametrization {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  static constexpr int DoF = 5;
  static constexpr int num_parameters = SE3t::num_parameters;

  using MatDiff = Eigen::Matrix<T, num_parameters, DoF>;
  using Tangent = Vec5t;

  SO3xS2Parametrization(const SE3 &f1ToWorld, const SE3 &f2ToWorld);
  SO3xS2Parametrization(const SO3t &baseRot, const Vec3t &centerTrans,
                        const Vec3t &initialTrans);

  MatDiff diffPlus() const;
  void addDelta(const Tangent &delta);

  inline SE3t value() const { return SE3t(mSo3.value(), mS2.value()); }
  inline const RightExpParametrization<SO3t> &so3() const { return mSo3; }
  inline RightExpParametrization<SO3t> &so3() { return mSo3; }
  inline const S2Parametrization &s2() const { return mS2; }
  inline S2Parametrization &s2() { return mS2; }

private:
  RightExpParametrization<SO3t> mSo3;
  S2Parametrization mS2;
};

class SO3xR3Parametrization {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  static constexpr int DoF = 6;
  static constexpr int num_parameters = SE3t::num_parameters;

  using MatDiff = Eigen::Matrix<T, num_parameters, DoF>;
  using Tangent = Vec6t;

  SO3xR3Parametrization(const SE3t &frameToWorld);

  MatDiff diffPlus() const;
  void addDelta(const Tangent &delta);

  inline SE3t value() const { return SE3t(mSo3.value(), mT); }
  inline const RightExpParametrization<SO3t> &so3() const { return mSo3; }
  inline RightExpParametrization<SO3t> &so3() { return mSo3; }
  inline const Vec3 &t() const { return mT; }
  inline Vec3 &t() { return mT; }

private:
  RightExpParametrization<SO3t> mSo3;
  Vec3t mT;
};

using SecondFrameParametrization = SO3xS2Parametrization;
#ifdef SO3_X_R3_PARAMETRIZATION
using FrameParametrization = SO3xR3Parametrization;
#else
using FrameParametrization = RightExpParametrization<SE3t>;
#endif

constexpr int sndDoF = SecondFrameParametrization::DoF;
constexpr int restDoF = FrameParametrization::DoF;
constexpr int affDoF = AffLightT::DoF;
constexpr int pointDoF = 1;

} // namespace mdso::optimize

#endif
