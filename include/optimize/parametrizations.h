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

private:
  void recalcOrts();

  Vec3t mValue;
  Vec3t center;
  T radius;
  Mat33t localToWorldRot;
};

class SO3xS2Parametrization {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  static constexpr int DoF = 5;
  static constexpr int num_parameters = SE3t::num_parameters;

  using MatDiff = Eigen::Matrix<T, num_parameters, DoF>;
  using Tangent = Vec5t;

  SO3xS2Parametrization(const SO3t &baseRot, const Vec3t &centerTrans,
                        const Vec3t &initialTrans);

  MatDiff diffPlus() const;

  void addDelta(const Tangent &delta);

  inline SE3t value() const { return SE3t(so3.value(), s2.value()); }

private:
  RightExpParametrization<SO3t> so3;
  S2Parametrization s2;
};

} // namespace mdso::optimize

#endif
