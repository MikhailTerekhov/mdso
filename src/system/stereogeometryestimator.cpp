#include "system/stereogeometryestimator.h"
#include <RelativePoseEstimator.h>
#include <ceres/ceres.h>

namespace fishdso {

StereoGeometryEstimator::StereoGeometryEstimator(
    CameraModel *cam, const std::vector<std::pair<Vec2, Vec2>> &imgCorresps)
    : cam(cam), imgCorresps(imgCorresps), rays(imgCorresps.size()),
      _depths(imgCorresps.size()), coarseFound(false), preciseFound(false),
      depthsEvaluated(false) {
  for (int i = 0; i < int(imgCorresps.size()); ++i) {
    rays[i].first = cam->unmap(imgCorresps[i].first.data()).normalized();
    rays[i].second = cam->unmap(imgCorresps[i].second.data()).normalized();
  }
}

const std::vector<int> &StereoGeometryEstimator::inliersInds() const {
  return _inliersInds;
}

EIGEN_STRONG_INLINE Vec2 calcDepthsInCorresp(
    const SE3 &motion, const std::pair<Vec3, Vec3> &rayCorresp) {
  Mat32 A;
  A.col(0) = motion.so3() * rayCorresp.first;
  A.col(1) = -rayCorresp.second;

  return A.fullPivHouseholderQr().solve(-motion.translation());
}

const std::vector<std::pair<double, double>> &
StereoGeometryEstimator::depths() {
  if (!depthsEvaluated) {
    depthsEvaluated = true;

    for (int i : _inliersInds) {
      Vec2 curDepths = calcDepthsInCorresp(motion, rays[i]);
      _depths[i].first = curDepths[0];
      _depths[i].second = curDepths[1];
    }
  }

  return _depths;
}

int StereoGeometryEstimator::inliersNum() { return _inliersInds.size(); }

int StereoGeometryEstimator::findInliersEssential(
    const Mat33 &E, std::vector<int> &inliersInds) {
  inliersInds.resize(0);
  int result = 0;
  Mat33 Et = E.transpose();
  Vec3 norm1, norm2;
  for (int i = 0; i < int(rays.size()); ++i) {
    auto r = rays[i];
    const auto &pr = imgCorresps[i];
    norm1 = Et * r.second;
    norm2 = E * r.first;
    if (norm1.squaredNorm() > 1e-4)
      r.first -= (r.first.dot(norm1) / norm1.squaredNorm()) * norm1;
    if (norm2.squaredNorm() > 1e-4)
      r.second -= (r.second.dot(norm2) / norm2.squaredNorm()) * norm2;

    double err1 = (cam->map(r.first.data()) - pr.first).squaredNorm();
    double err2 = (cam->map(r.second.data()) - pr.second).squaredNorm();

    if (std::min(err1, err2) < settingEssentialReprojErrThreshold) {
      ++result;
      inliersInds.push_back(i);
    }
  }
  return result;
}

int StereoGeometryEstimator::findInliersMotion(const SE3 &motion,
                                               std::vector<int> &inliersInds) {
  static std::vector<int> newInlierInds = reservedVector();

  newInlierInds.resize(0);
  for (int i : inliersInds) {
    Vec2 depths = calcDepthsInCorresp(motion, rays[i]);
    if (depths[0] > 0 && depths[1] > 0)
      newInlierInds.push_back(i);
  }
  std::swap(newInlierInds, inliersInds);

  return inliersInds.size();
}

SE3 StereoGeometryEstimator::extractMotion(const Mat33 &E,
                                           std::vector<int> &inliersInds,
                                           int &newInliers) {
  Eigen::JacobiSVD<Mat33> svdE(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Mat33 U = svdE.matrixU();
  Mat33 V = svdE.matrixV();

  U *= U.determinant();
  V *= V.determinant();

  Vec3 t = U.block<3, 1>(0, 2);
  Mat33 W;
  // clang-format off
  W << 0, -1, 0,
       1,  0, 0,
       0,  0, 1;
  // clang-format on

  Mat33 R1 = U * W * V.transpose();
  Mat33 R2 = U * W.transpose() * V.transpose();

  SE3 solutions[4] = {SE3(R1, t), SE3(R1, -t), SE3(R2, t), SE3(R2, -t)};

  //  Mat33 tCross;
  //  tCross << 0, -t[2], t[1], t[2], 0, -t[0], -t[1], t[0], 0;

  int bestFrontPointsNum = 0;
  SE3 bestSol;

  static std::vector<int> bestInliersInds = reservedVector(),
                          curInliersInds = reservedVector();
  bestInliersInds.resize(0);
  curInliersInds.resize(0);

  for (SE3 sol : solutions) {
    curInliersInds = inliersInds;
    int curFrontPointsNum = findInliersMotion(sol, curInliersInds);

    if (curFrontPointsNum > bestFrontPointsNum) {
      bestFrontPointsNum = curFrontPointsNum;
      bestSol = sol;
      std::swap(curInliersInds, bestInliersInds);
    }
  }
  std::swap(inliersInds, bestInliersInds);
  newInliers = bestFrontPointsNum;
  return bestSol;
}

SE3 StereoGeometryEstimator::findCoarseMotion() {
  if (coarseFound || preciseFound)
    return motion;

  static relative_pose::GeneralizedCentralRelativePoseEstimator<double> est;
  constexpr int N = settingEssentialMinimalSolveN;
  const double p = settingEssentialSuccessProb;

  SE3 bestMotion;

  std::mt19937 mt;
  std::uniform_int_distribution<> inds(0, rays.size() - 1);

  int hypotesisInd[N];
  std::pair<Vec3 *, Vec3 *> hypotesis[N];
  Mat33 results[10];
  int bestInliers = -1;
  long long iterNum = settingRansacMaxIter;
  double q = std::pow(1.0 - std::pow(1 - p, 1.0 / iterNum), 1.0 / N);

  for (int it = 0; it < iterNum; ++it) {
    for (int i = 0; i < N; ++i)
      hypotesisInd[i] = inds(mt);
    std::sort(hypotesisInd, hypotesisInd + N);
    bool isRepeated = false;
    for (int i = 0; i + 1 < N; ++i)
      if (hypotesisInd[i] == hypotesisInd[i + 1])
        isRepeated = true;
    if (isRepeated)
      continue;
    for (int i = 0; i < N; ++i)
      hypotesis[i] = std::make_pair<Vec3 *, Vec3 *>(
          &rays[hypotesisInd[i]].second, &rays[hypotesisInd[i]].first);

    int foundN = est.estimate(hypotesis, N, results);
    int maxInliers = 0, maxInliersInd = 0;

    //    for (int i = 0; i < foundN; ++i) {

    //      std::cout << "ok, errors on input vectors for E" << i << ": ";
    //      for (int j = 0; j < N; ++j)
    //        std::cout << std::abs(hypotesis[j].second->transpose() *
    //        results[i] *
    //                              (*hypotesis[j].first))
    //                  << ' ';
    //      std::cout << std::endl;
    //    }

    //    std::cout << "now same stuff but E is transposed:\n";

    //    for (int i = 0; i < foundN; ++i) {

    //      std::cout << "ok, errors on input vectors for E" << i << ": ";
    //      for (int j = 0; j < N; ++j)
    //        std::cout << std::abs(hypotesis[j].second->transpose() *
    //                              results[i].transpose() *
    //                              (*hypotesis[j].first))
    //                  << ' ';
    //      std::cout << std::endl;
    //    }

    //    std::cout << "resulting norms = ";
    //    for (int i = 0; i < foundN; ++i)
    //      std::cout << results[i].norm() << ' ';
    //    std::cout << std::endl;

    static std::vector<int> curInliersInds = reservedVector(),
                            bestInliersInds = reservedVector();
    for (int i = 0; i < foundN; ++i) {
      int inliers = findInliersEssential(results[i], curInliersInds);
      if (inliers > maxInliers) {
        maxInliers = inliers;
        maxInliersInd = i;
        std::swap(bestInliersInds, curInliersInds);
      }
    }
    SE3 curMotion =
        extractMotion(results[maxInliersInd], bestInliersInds, maxInliers);
    if (bestInliers < maxInliers) {
      bestInliers = maxInliers;
      bestMotion = curMotion;
      std::swap(_inliersInds, bestInliersInds);
    }

    double curQ = double(maxInliers) / rays.size();
    if (curQ > q) {
      q = curQ;
      double newIterNum = std::log(1 - p) / std::log(1 - std::pow(curQ, N));
      iterNum = static_cast<long long>(newIterNum);
    }
  }

  std::cout << "iterNum = " << iterNum << std::endl;
  coarseFound = true;
  motion = bestMotion;

  return bestMotion;
}

class SphericalPlus {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  SphericalPlus(const Vec3 &_k) : k(_k.normalized()) {
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

    Mat33t R;
    if (rotAxisSqN < 1e-4)
      R = degenerateR.cast<T>();
    else
      R = -Mat33t::Identity() +
          ((T(2.0) / rotAxisSqN) * rotAxis * rotAxis.transpose());

    Vec3t resV = (vec + R * kDeltaOrts * delta).normalized();
    memcpy(res, resV.data(), 3 * sizeof(T));

    return true;
  }

private:
  Mat32 kDeltaOrts;
  Vec3 k;
  static Mat33 degenerateR;
};

// clang-format off
Mat33 SphericalPlus::degenerateR =
    (Mat33() << 1.0,  0.0,  0.0,
                0.0, -1.0,  0.0,
                0.0,  0.0, -1.0).finished();
// clang-format on

SE3 StereoGeometryEstimator::findPreciseMotion() {
  if (preciseFound)
    return motion;
  if (!coarseFound)
    findCoarseMotion();

  //  std::vector<int> inlierInds;
  //  inlierInds.reserve(inliersNum);
  //  std::cout << "inliers before cleanup = " << inliersNum << std::endl;
  //  for (int i = 0; i < inliersMask.size(); ++i)
  //    if (inliersMask[i])
  //      inlierInds.push_back(i);
  //  std::cout << "inliers in inds = " << inlierInds.size() << std::endl;

  //  std::sort(inlierInds.begin(), inlierInds.end(), [&](int i1, int i2) {
  //    ReprojectionResidual res1(cam, raysNorm[i1].first, raysNorm[i1].second,
  //                              imgCorresps[i1].first,
  //                              imgCorresps[i1].second);
  //    ReprojectionResidual res2(cam, raysNorm[i2].first, raysNorm[i2].second,
  //                              imgCorresps[i2].first,
  //                              imgCorresps[i2].second);
  //    double results1[4], results2[4];
  //    res1(motion.unit_quaternion().coeffs().data(),
  //    motion.translation().data(),
  //         results1);
  //    res2(motion.unit_quaternion().coeffs().data(),
  //    motion.translation().data(),
  //         results2);

  //    double norm1 = std::hypot(std::hypot(results1[0], results1[1]),
  //                              std::hypot(results1[2], results1[3]));
  //    double norm2 = std::hypot(std::hypot(results2[0], results2[1]),
  //                              std::hypot(results2[2], results2[3]));
  //    return norm1 < norm2;
  //  });

  //  std::cout << inlierInds.size() << ' ' << settingRemoveResidualsRatio << '
  //  '
  //            << int(inlierInds.size() * settingRemoveResidualsRatio)
  //            << std::endl;

  //  int toRemove = int(inlierInds.size() * settingRemoveResidualsRatio);
  //  for (int i = 0; i < toRemove; ++i)
  //    inlierInds.pop_back();
  //  std::cout << inlierInds.size() << std::endl;

  //  std::fill(inliersMask.begin(), inliersMask.end(), 0);
  //  for (int i = 0; i < inlierInds.size(); ++i)
  //    inliersMask[inlierInds[i]] = true;
  //  inliersNum = inlierInds.size();

  //  std::cout << "corresps in optim = " << inliersNum << std::endl;

  ceres::Problem problem;

  problem.AddParameterBlock(motion.so3().data(), 4,
                            new ceres::EigenQuaternionParameterization());
  problem.AddParameterBlock(
      motion.translation().data(), 3,
      new ceres::AutoDiffLocalParameterization<SphericalPlus, 3, 2>(
          new SphericalPlus(motion.translation())));

  for (int i : _inliersInds)
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<ReprojectionResidual, 4, 4, 3>(
            new ReprojectionResidual(cam, rays[i].first, rays[i].second,
                                     imgCorresps[i].first,
                                     imgCorresps[i].second)),
        new ceres::ArctanLoss(1), motion.so3().data(),
        motion.translation().data());

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  // options.minimizer_progress_to_stdout = true;
  options.max_num_iterations = 10;
  ceres::Solver::Summary summary;

  ceres::Solve(options, &problem, &summary);

  // std::cout << summary.FullReport() << "\n";
  //  std::cout << "avg residual before = "
  //            << std::sqrt(summary.initial_cost / summary.num_residuals)
  //            << "\navg residual after = "
  //            << std::sqrt(summary.final_cost / summary.num_residuals)
  //            << std::endl;

  findInliersEssential(SO3::hat(motion.translation()) * motion.rotationMatrix(),
                       _inliersInds);
  findInliersMotion(motion, _inliersInds);

  preciseFound = true;
  return motion;
}

} // namespace fishdso
