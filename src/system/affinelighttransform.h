#pragma once

#include <Eigen/Core>

namespace fishdso {

template <typename T> struct AffineLightTransform {
  AffineLightTransform() : data{1.0, 0.0} {}
  AffineLightTransform(T a, T b) : data{a, b} {}

  EIGEN_STRONG_INLINE T operator()(T x) { return data[0] * x + data[1]; }

  T data[2];
};

} // namespace fishdso
