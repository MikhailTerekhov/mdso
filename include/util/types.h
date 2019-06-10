#ifndef INCLUDE_TYPES
#define INCLUDE_TYPES

#include <Eigen/Core>
#include <Eigen/StdVector>
#include <map>
#include <memory>
#include <queue>
#include <sophus/se3.hpp>
#include <sophus/sim3.hpp>
#include <unordered_set>
#include <utility>
#include <vector>

namespace fishdso {

typedef Eigen::Matrix<double, 2, 1> Vec2;
typedef Eigen::Matrix<double, 3, 1> Vec3;
typedef Eigen::Matrix<double, 4, 1> Vec4;
typedef Eigen::Matrix<double, 5, 1> Vec5;
typedef Eigen::Matrix<double, 9, 1> Vec9;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> VecX;

typedef Eigen::Matrix<int, 2, 1> Vec2i;

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

typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> MatXXi;

typedef Eigen::Quaterniond Quaternion;

typedef Sophus::Sim3d Sim3;
typedef Sophus::SE3d SE3;
typedef Sophus::SO3d SO3;

template <typename T>
using StdVector = std::vector<T, Eigen::aligned_allocator<T>>;

template <typename T>
using StdQueue = std::queue<T, std::deque<T, Eigen::aligned_allocator<T>>>;

template <typename K, typename T>
using StdMap =
    std::map<K, T, std::less<K>, Eigen::aligned_allocator<std::pair<K, T>>>;

class OptionalDeleter {
  bool doDelete;

public:
  OptionalDeleter(bool newDoDelete = true)
      : doDelete(newDoDelete) {}
  template <typename T> void operator()(T *p) const {
    if (doDelete)
      delete p;
  }
};

template <typename T> using SetUniquePtr = std::unique_ptr<T, OptionalDeleter>;

template <typename T> SetUniquePtr<T> makeFindPtr(T *ptr) {
  return SetUniquePtr<T>(ptr, false);
}

template <typename T>
using StdUnorderedSetOfPtrs = std::unordered_set<SetUniquePtr<T>>;

} // namespace fishdso

#endif
