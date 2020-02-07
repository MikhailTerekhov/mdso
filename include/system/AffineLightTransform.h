#ifndef INCLUDE_AFFINELIGHTTRANSFORM
#define INCLUDE_AFFINELIGHTTRANSFORM

#include <Eigen/Core>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <ostream>
#include <random>

namespace mdso {

template <typename T> struct AffineLightTransform {
  static constexpr int DoF = 2;

  AffineLightTransform()
      : data{T(0.0), T(0.0)} {}
  AffineLightTransform(const T &a, const T &b)
      : data{a, b} {}

  AffineLightTransform<T> &operator=(const AffineLightTransform<T> &other) {
    memcpy(data, other.data, 2 * sizeof(T));
    return *this;
  }

  inline T operator()(const T &x, const T &expA) const {
    return expA * x + b();
  }
  inline T operator()(const T &x) const { return (*this)(x, ea()); }
  template <typename U> cv::Mat_<U> operator()(const cv::Mat_<U> &mat) const {
    cv::Mat_<U> result;
    cv::convertScaleAbs(mat, result, ea(), b());
    return result;
  }

  inline T a() const { return data[0]; }
  inline T ea() const { return exp(data[0]); }
  inline T b() const { return data[1]; }

  friend AffineLightTransform<T>
  operator*(const AffineLightTransform<T> &first,
            const AffineLightTransform<T> &second) {
    return AffineLightTransform<T>(first.a() + second.a(),
                                   first.ea() * second.b() + first.b());
  }

  AffineLightTransform<T> inverse() const {
    return AffineLightTransform<T>(-a(), -b() * exp(-a()));
  }

  static void normalizeMultiplier(AffineLightTransform<T> &toNormalize,
                                  AffineLightTransform<T> &relative) {
    relative.data[0] -= toNormalize.data[0];
    toNormalize.data[0] = T(0);
  }

  template <typename U> AffineLightTransform<U> cast() const {
    return AffineLightTransform<U>(U(data[0]), U(data[1]));
  }

  friend std::ostream &operator<<(std::ostream &os,
                                  const AffineLightTransform<T> &affLight) {
    os << "raw     = " << affLight.data[0] << ' ' << affLight.data[1]
       << "\nas ax+b = " << affLight.ea() << ' ' << affLight.b() << '\n';
    return os;
  }

  T data[DoF];
};

} // namespace mdso

#endif
