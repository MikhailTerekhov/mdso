#ifndef INCLUDE_TYPES
#define INCLUDE_TYPES

#include "system/AffineLightTransform.h"
#include <Eigen/Core>
#include <Eigen/StdVector>
#include <boost/container/static_vector.hpp>
#include <boost/multi_array.hpp>
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

namespace optimize {
using T = OPT_T;
using Vec2t = Eigen::Matrix<T, 2, 1>;
using Vec3t = Eigen::Matrix<T, 3, 1>;
using Vec4t = Eigen::Matrix<T, 4, 1>;
using Vec5t = Eigen::Matrix<T, 5, 1>;
using Vec6t = Eigen::Matrix<T, 6, 1>;
using Vec7t = Eigen::Matrix<T, 7, 1>;
using Vec8t = Eigen::Matrix<T, 8, 1>;
using Vec9t = Eigen::Matrix<T, 9, 1>;
using VecXt = Eigen::Matrix<T, Eigen::Dynamic, 1>;

using Mat22t = Eigen::Matrix<T, 2, 2>;
using Mat23t = Eigen::Matrix<T, 2, 3>;
using Mat24t = Eigen::Matrix<T, 2, 4>;
using Mat26t = Eigen::Matrix<T, 2, 6>;
using Mat27t = Eigen::Matrix<T, 2, 7>;
using Mat33t = Eigen::Matrix<T, 3, 3>;
using Mat34t = Eigen::Matrix<T, 3, 4>;
using Mat36t = Eigen::Matrix<T, 3, 6>;
using Mat37t = Eigen::Matrix<T, 3, 7>;
using Mat43t = Eigen::Matrix<T, 4, 3>;
using Mat44t = Eigen::Matrix<T, 4, 4>;
using Mat62t = Eigen::Matrix<T, 6, 2>;
using Mat72t = Eigen::Matrix<T, 7, 2>;
using Mat75t = Eigen::Matrix<T, 7, 5>;
using Mat76t = Eigen::Matrix<T, 7, 6>;
using Mat77t = Eigen::Matrix<T, 7, 7>;
using Mat88t = Eigen::Matrix<T, 8, 8>;
using Mat99t = Eigen::Matrix<T, 9, 9>;
using Mat12x3t = Eigen::Matrix<T, 12, 3>;
using Mat12x4t = Eigen::Matrix<T, 12, 4>;
using MatX2t = Eigen::Matrix<T, Eigen::Dynamic, 2>;
using MatX3t = Eigen::Matrix<T, Eigen::Dynamic, 3>;
using MatX4t = Eigen::Matrix<T, Eigen::Dynamic, 4>;
using MatXXt = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

using SE3t = Sophus::SE3<T>;
using SO3t = Sophus::SO3<T>;

using AffLightT = AffineLightTransform<T>;
} // namespace optimize

using Vec2 = Eigen::Matrix<double, 2, 1>;
using Vec3 = Eigen::Matrix<double, 3, 1>;
using Vec4 = Eigen::Matrix<double, 4, 1>;
using Vec5 = Eigen::Matrix<double, 5, 1>;
using Vec6 = Eigen::Matrix<double, 6, 1>;
using Vec8 = Eigen::Matrix<double, 8, 1>;
using Vec9 = Eigen::Matrix<double, 9, 1>;
using VecX = Eigen::Matrix<double, Eigen::Dynamic, 1>;

using Vec2i = Eigen::Matrix<int, 2, 1>;

using Mat22 = Eigen::Matrix<double, 2, 2>;
using Mat23 = Eigen::Matrix<double, 2, 3>;
using Mat32 = Eigen::Matrix<double, 3, 2>;
using Mat33 = Eigen::Matrix<double, 3, 3>;
using Mat34 = Eigen::Matrix<double, 3, 4>;
using Mat37 = Eigen::Matrix<double, 3, 7>;
using Mat43 = Eigen::Matrix<double, 4, 3>;
using Mat44 = Eigen::Matrix<double, 4, 4>;
using Mat55 = Eigen::Matrix<double, 5, 5>;
using Mat77 = Eigen::Matrix<double, 7, 7>;
using MatX5 = Eigen::Matrix<double, Eigen::Dynamic, 5>;
using MatX9 = Eigen::Matrix<double, Eigen::Dynamic, 9>;
using MatXX = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

using MatXXi = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic>;

using Quaternion = Eigen::Quaterniond;

using Sim3 = Sophus::Sim3d;
using SE3 = Sophus::SE3d;
using SO3 = Sophus::SO3d;

using AffLight = AffineLightTransform<double>;

using Vec2f = Eigen::Matrix<float, 2, 1>;
using Vec3f = Eigen::Matrix<float, 3, 1>;
using Mat23f = Eigen::Matrix<float, 2, 3>;

namespace fs = std::filesystem;

using boost::container::static_vector;
template <typename T> using Array2d = boost::multi_array<T, 2>;
template <typename T> using Array3d = boost::multi_array<T, 3>;
template <typename T> using Array4d = boost::multi_array<T, 4>;

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
using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;

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
                                      StdVector<std::pair<Timestamp, SE3>>,
                                      TimestampPoseComp>;

} // namespace mdso

#endif
