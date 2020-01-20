#ifndef INCLUDE_SE3BASEANDTANGENT
#define INCLUDE_SE3BASEANDTANGENT

#include "util/types.h"

namespace mdso {

template <typename LieGroup> class BaseAndTangent {
  using Tangent = typename LieGroup::Tangent;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  BaseAndTangent(const LieGroup &init)
      : base(init)
      , mDelta(LieGroup::Tangent::Zero())
      , combined(init) {}

  BaseAndTangent(const LieGroup &base, const Tangent &delta)
      : base(base)
      , mDelta(delta)
      , combined(LieGroup::exp(delta) * base) {}

  inline const LieGroup &operator()() const { return combined; }

  inline void setValue(const LieGroup &newValue) {
    base = newValue;
    combined = newValue;
    mDelta = LieGroup::Tangent::Zero();
  }

  inline const Tangent &delta() const { return mDelta; }
  inline void setDelta(const Tangent &newDelta) {
    mDelta = newDelta;
    combined = LieGroup::exp(mDelta) * base;
  }

  inline bool isTangentFixed() const { return mIsTangentFixed; }
  inline void fixTangent() { mIsTangentFixed = true; }

  void updateTangent() {
    CHECK(!isTangentFixed());
    base = combined;
    mDelta = LieGroup::Tangent::Zero();
  }

private:
  // invariant: combined = SE3::exp(delta) * base
  LieGroup base;
  Tangent mDelta;
  LieGroup combined;
  bool mIsTangentFixed;
};

using SE3BaseAndTangent = BaseAndTangent<SE3>;

} // namespace mdso

#endif
