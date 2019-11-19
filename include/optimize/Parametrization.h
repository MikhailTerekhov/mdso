#ifndef INCLUDE_SE3PARAMETRIZATION
#define INCLUDE_SE3PARAMETRIZATION

#include "util/BaseAndTangent.h"
#include "util/types.h"
#include <variant>

namespace mdso::optimize {

template <typename LieGroup> struct LeftExpParametrization {
  static constexpr int DoF = LieGroup::DoF;

  using MatDiff = Eigen::Matrix<T, LieGroup::num_parameters, DoF>;
  using Tangent = typename LieGroup::Tangent;

  LeftExpParametrization(const BaseAndTangent<LieGroup> &baseAndTangent)
      : baseAndTangent(baseAndTangent) {}

  MatDiff diffPlus() {
    // TODO
    return MatDiff::Zero();
  }

  BaseAndTangent<LieGroup> baseAndTangent;
};

LeftExpParametrization<SE3t>
castSE3Parametrization(const BaseAndTangent<SE3> &other) {
  return LeftExpParametrization<SE3t>(BaseAndTangent<SE3t>(other().cast<T>()));
}

struct S2Parametrization {
  static constexpr int num_parameters = 3;
  static constexpr int DoF = 2;

  using MatDiff = Eigen::Matrix<T, num_parameters, DoF>;
  using Tangent = Vec2t;

  S2Parametrization(const Vec3t &center, T radius) {
    // TODO
  }

  S2Parametrization(const SE3t &expDelta) {
    // TODO
  }

  MatDiff diffPlus() {
    // TODO
    return MatDiff::Zero();
  }

  Vec3t base;
  Tangent delta;
};

struct SO3xS2Parametrization {
  static constexpr int DoF = 5;

  using MatDiff = Eigen::Matrix<T, SE3t::num_parameters, DoF>;
  using Tangent = Vec5t;

  SO3xS2Parametrization(const BaseAndTangent<SE3> &baseAndTangent)
      : so3(BaseAndTangent<SO3t>(baseAndTangent().so3(),
                                 baseAndTangent.delta().tail<3>()))
      , s2(SE3t::exp(baseAndTangent.delta())) {}

  static MatDiff diffPlus(const SE3t &base, const Tangent &delta) {
    // TODO
    return MatDiff::Zero();
  }

  LeftExpParametrization<SO3t> so3;
  S2Parametrization s2;
};

struct ConstParametrization {
  static constexpr int DoF = 0;
};

} // namespace mdso::optimize

#endif
