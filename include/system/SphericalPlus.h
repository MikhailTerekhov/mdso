#ifndef INCLUDE_SPHERICALPLUS
#define INCLUDE_SPHERICALPLUS

#include "util/types.h"
#include <Eigen/Core>
#include <ceres/ceres.h>

namespace fishdso {

class SphericalPlus {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  SphericalPlus(const Vec3 &center, double radius, const Vec3 &initialValue)
      : center(center), radius(radius),
        k((initialValue - center).normalized()) {
    CHECK(((initialValue - center).norm() - radius) / radius < 1e-4);
    int minI = std::min_element(k.data(), k.data() + 3,
                                [](double a, double b) {
                                  return std::abs(a) < std::abs(b);
                                }) -
               k.data();
    Vec3 v1, v2;
    if (minI == 0)
      v1 = Vec3(0, -k[2], k[1]).normalized();
    else if (minI == 1)
      v1 = Vec3(-k[2], 0, k[0]).normalized();
    else
      v1 = Vec3(-k[1], k[0], 0).normalized();
    v2 = k.cross(v1).normalized();
    kDeltaOrts.col(0) = v1;
    kDeltaOrts.col(1) = v2;
  }

  template <typename T>
  bool operator()(const T *const vecP, const T *const deltaP, T *res) const {
    typedef Eigen::Matrix<T, 3, 1> Vec3t;
    typedef Eigen::Matrix<T, 2, 1> Vec2t;
    typedef Eigen::Matrix<T, 3, 2> Mat32t;
    typedef Eigen::Matrix<T, 3, 3> Mat33t;

    Vec3t kT = k.cast<T>();
    Eigen::Map<const Vec3t> vecM(vecP);
    Vec3t vec = vecM;
    Eigen::Map<const Vec2t> deltaM(deltaP);
    Vec2t delta = deltaM;
    Vec3t rotAxis = vec + k;
    T rotAxisSqN = rotAxis.squaredNorm();

    vec -= center.cast<T>();

    Mat33t R;
    if (rotAxisSqN < 1e-4)
      R = degenerateR.cast<T>();
    else
      R = -Mat33t::Identity() +
          ((T(2.0) / rotAxisSqN) * rotAxis * rotAxis.transpose());

    Vec3t resV = (vec + R * kDeltaOrts * delta).normalized() * T(radius) +
                 center.cast<T>();
    memcpy(res, resV.data(), 3 * sizeof(T));

    return true;
  }

private:
  Mat32 kDeltaOrts;
  Vec3 center;
  double radius;
  Vec3 k;
  static Mat33 degenerateR;
};

} // namespace fishdso

#endif
