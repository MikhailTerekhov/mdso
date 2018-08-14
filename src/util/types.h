#pragma once

#include <Eigen/Core>

namespace fishdso {

typedef Eigen::Matrix<double, 3, 1> Vec3;
typedef Eigen::Matrix<double, 2, 1> Vec2;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> VecX;

typedef Eigen::Matrix<double, 3, 3> Mat33;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatXX;

} // namespace fishdso
