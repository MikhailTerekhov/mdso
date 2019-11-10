#ifndef INCLUDE_TYPES
#define INCLUDE_TYPES

#include "system/AffineLightTransform.h"
#include <Eigen/Core>
#include <Eigen/StdVector>
#include <boost/container/static_vector.hpp>
#include <filesystem>
#include <map>
#include <memory>
#include <queue>
#include <sophus/se3.hpp>
#include <sophus/sim3.hpp>
#include <unordered_set>
#include <utility>
#include <vector>

namespace mdso {

using Vec2 = Eigen::Matrix<double, 2, 1>;
using Vec3 = Eigen::Matrix<double, 3, 1>;
using Vec4 = Eigen::Matrix<double, 4, 1>;
using Vec5 = Eigen::Matrix<double, 5, 1>;
using Vec8 = Eigen::Matrix<double, 8, 1>;
using Vec9 = Eigen::Matrix<double, 9, 1>;
using VecX = Eigen::Matrix<double, Eigen::Dynamic, 1>;

using Vec2i = Eigen::Matrix<int, 2, 1>;

using Mat22 = Eigen::Matrix<double, 2, 2>;
using Mat23 = Eigen::Matrix<double, 2, 3>;
using Mat32 = Eigen::Matrix<double, 3, 2>;
using Mat33 = Eigen::Matrix<double, 3, 3>;
using Mat34 = Eigen::Matrix<double, 3, 4>;
using Mat43 = Eigen::Matrix<double, 4, 3>;
using Mat44 = Eigen::Matrix<double, 4, 4>;
using Mat55 = Eigen::Matrix<double, 5, 5>;
using MatX5 = Eigen::Matrix<double, Eigen::Dynamic, 5>;
using MatX9 = Eigen::Matrix<double, Eigen::Dynamic, 9>;
using MatXX = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

using MatXXi = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic>;

using Quaternion = Eigen::Quaterniond;

using Sim3 = Sophus::Sim3d;
using SE3 = Sophus::SE3d;
using SO3 = Sophus::SO3d;

using AffLight = AffineLightTransform<double>;

namespace fs = std::filesystem;

using boost::container::static_vector;

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

using Timestamp = int64_t;

template <typename T>
using StdUnorderedSetOfPtrs = std::unordered_set<SetUniquePtr<T>>;

class TimestampPoseComp {
public:
  bool operator()(const std::pair<Timestamp, SE3> &a,
                  const std::pair<Timestamp, SE3> &b) {
    return a.first > b.first;
  }
};

// used to store poses in trajectory writers
using PosesPool = std::priority_queue<std::pair<Timestamp, SE3>,
                                      std::vector<std::pair<Timestamp, SE3>>,
                                      TimestampPoseComp>;

} // namespace mdso

#endif
