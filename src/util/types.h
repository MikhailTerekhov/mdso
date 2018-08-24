#pragma once

#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <sophus/sim3.hpp>

namespace fishdso {

typedef Eigen::Matrix<double, 2, 1> Vec2;
typedef Eigen::Matrix<double, 3, 1> Vec3;
typedef Eigen::Matrix<double, 5, 1> Vec5;
typedef Eigen::Matrix<double, 9, 1> Vec9;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> VecX;

typedef Eigen::Matrix<double, 2, 2> Mat22;
typedef Eigen::Matrix<double, 2, 3> Mat23;
typedef Eigen::Matrix<double, 3, 2> Mat32;
typedef Eigen::Matrix<double, 3, 3> Mat33;
typedef Eigen::Matrix<double, 5, 5> Mat55;
typedef Eigen::Matrix<double, Eigen::Dynamic, 5> MatX5;
typedef Eigen::Matrix<double, Eigen::Dynamic, 9> MatX9;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatXX;

typedef Eigen::Quaterniond Quaternion;

typedef Sophus::Sim3d Sim3;
typedef Sophus::SE3d SE3;
typedef Sophus::SO3d SO3;

} // namespace fishdso
