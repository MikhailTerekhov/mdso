#pragma once

#include <Eigen/Core>
#include <Eigen/StdVector>
#include <map>
#include <sophus/se3.hpp>
#include <sophus/sim3.hpp>
#include <utility>
#include <vector>

namespace fishdso {

typedef Eigen::Matrix<double, 2, 1> Vec2;
typedef Eigen::Matrix<double, 3, 1> Vec3;
typedef Eigen::Matrix<double, 4, 1> Vec4;
typedef Eigen::Matrix<double, 5, 1> Vec5;
typedef Eigen::Matrix<double, 9, 1> Vec9;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> VecX;

typedef Eigen::Matrix<double, 2, 2> Mat22;
typedef Eigen::Matrix<double, 2, 3> Mat23;
typedef Eigen::Matrix<double, 3, 2> Mat32;
typedef Eigen::Matrix<double, 3, 3> Mat33;
typedef Eigen::Matrix<double, 3, 4> Mat34;
typedef Eigen::Matrix<double, 4, 3> Mat43;
typedef Eigen::Matrix<double, 4, 4> Mat44;
typedef Eigen::Matrix<double, 5, 5> Mat55;
typedef Eigen::Matrix<double, Eigen::Dynamic, 5> MatX5;
typedef Eigen::Matrix<double, Eigen::Dynamic, 9> MatX9;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatXX;

typedef Eigen::Quaterniond Quaternion;

typedef Sophus::Sim3d Sim3;
typedef Sophus::SE3d SE3;
typedef Sophus::SO3d SO3;

typedef std::vector<Vec2, Eigen::aligned_allocator<Vec2>> stdvectorVec2;
typedef std::vector<SE3, Eigen::aligned_allocator<SE3>> stdvectorSE3;
typedef std::vector<SO3, Eigen::aligned_allocator<SO3>> stdvectorSO3;
typedef std::vector<std::pair<Vec2, Vec2>,
                    Eigen::aligned_allocator<std::pair<Vec2, Vec2>>>
    stdvectorStdpairVec2Vec2;
typedef std::vector<std::pair<Vec2, double>,
                    Eigen::aligned_allocator<std::pair<Vec2, double>>>
    stdvectorStdpairVec2double;

typedef std::map<int, SE3, std::less<int>,
                 Eigen::aligned_allocator<std::pair<int, SE3>>>
    stdmapIntSE3;

} // namespace fishdso
