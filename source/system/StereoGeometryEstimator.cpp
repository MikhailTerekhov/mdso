#include "system/StereoGeometryEstimator.h"
#include "system/SphericalPlus.h"
#include "util/geometry.h"
#include <RelativePoseEstimator.h>
#include <ceres/ceres.h>
#include <fstream>
#include <glog/logging.h>

namespace fishdso {

const int inlierVectorsUsed = 5;
const int motionInliers = 1;
const int extractBestInliers = 2;
const int extractCurInliers = 3;
const int ransacBestInliers = 4;
const int ransacCurInliers = 5;

StereoGeometryEstimator::StereoGeometryEstimator(
    CameraModel *cam, const StdVector<std::pair<Vec2, Vec2>> &imgCorresps)
    : cam(cam), imgCorresps(imgCorresps), rays(imgCorresps.size()),
      _depths(imgCorresps.size()),
      inlierVectorsPool(reservedVector<int>(settingKeyPointsCount),
                        inlierVectorsUsed),
      coarseFound(false), preciseFound(false), depthsEvaluated(false) {
  for (int i = 0; i < int(imgCorresps.size()); ++i) {
    rays[i].first = cam->unmap(imgCorresps[i].first.data()).normalized();
    rays[i].second = cam->unmap(imgCorresps[i].second.data()).normalized();
  }
}

const std::vector<int> &StereoGeometryEstimator::inliersInds() const {
  return _inliersInds;
}

const std::vector<std::pair<double, double>> &
StereoGeometryEstimator::depths() {
  if (!depthsEvaluated) {
    depthsEvaluated = true;

    for (int i : _inliersInds) {
      Vec2 curDepths = triangulate(motion, rays[i].first, rays[i].second);
      _depths[i].first = curDepths[0];
      _depths[i].second = curDepths[1];
    }

    // int neededInd = *std::min_element(
    // _inliersInds.begin(), _inliersInds.end(), [this](int i1, int i2) {
    // if (imgCorresps[i1].second[0] <= 600 &&
    // imgCorresps[i2].second[0] > 600)
    // return true;
    // else if (imgCorresps[i1].second[0] > 600 &&
    // imgCorresps[i2].second[0] <= 600)
    // return false;

    // return imgCorresps[i1].second[1] > imgCorresps[i2].second[1];
    // });
    // Vec2 unm = cam->map(rays[neededInd].second);
    // std::cout << "min depth position = " << unm.transpose();
  }

  return _depths;
}

void StereoGeometryEstimator::outputInlierCorresps() {
  std::ofstream out(FLAGS_output_directory + "/corresps.txt");
  for (int i : _inliersInds) {
    out << i << std::endl;
    out << rays[i].first.transpose() << std::endl;
    out << rays[i].second.transpose() << std::endl;
    out << motion.translation().transpose() << std::endl;
    out << motion.unit_quaternion().coeffs().transpose() << std::endl;
  }
  out.close();
}

int StereoGeometryEstimator::inliersNum() { return _inliersInds.size(); }

double reprojectionError(CameraModel *cam, const Mat33 &E, const Mat33 &Et,
                         const std::pair<Vec2, Vec2> &imgCorresp,
                         std::pair<Vec3, Vec3> rayCorresp) {
  Vec3 norm1 = Et * rayCorresp.second;
  Vec3 norm2 = E * rayCorresp.first;
  if (norm1.squaredNorm() > 1e-4)
    rayCorresp.first -=
        (rayCorresp.first.dot(norm1) / norm1.squaredNorm()) * norm1;
  if (norm2.squaredNorm() > 1e-4)
    rayCorresp.second -=
        (rayCorresp.second.dot(norm2) / norm2.squaredNorm()) * norm2;

  double err1 = (cam->map(rayCorresp.first.data()) - imgCorresp.first).norm();
  double err2 = (cam->map(rayCorresp.second.data()) - imgCorresp.second).norm();

  return std::min(err1, err2);
}

int StereoGeometryEstimator::findInliersEssential(
    const Mat33 &E, std::vector<int> &inliersInds) {
  inliersInds.resize(0);
  int result = 0;
  Mat33 Et = E.transpose();
  for (int i = 0; i < int(rays.size()); ++i) {
    if (reprojectionError(cam, E, Et, imgCorresps[i], rays[i]) <
        settingEssentialReprojErrThreshold) {
      ++result;
      inliersInds.push_back(i);
    }
  }
  return result;
}

int StereoGeometryEstimator::findInliersMotion(const SE3 &motion,
                                               std::vector<int> &inliersInds) {
  std::vector<int> &newInlierInds = inlierVectorsPool.get(motionInliers);

  newInlierInds.resize(0);
  for (int i : inliersInds) {
    Vec2 depths = triangulate(motion, rays[i].first, rays[i].second);
    if (depths[0] > 0 && depths[1] > 0)
      newInlierInds.push_back(i);
  }
  std::swap(newInlierInds, inliersInds);

  return inliersInds.size();
}

SE3 StereoGeometryEstimator::extractMotion(const Mat33 &E,
                                           std::vector<int> &inliersInds,
                                           int &newInliers,
                                           bool doLogFrontPoints) {
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

  std::vector<int> bestInliersInds = inlierVectorsPool.get(extractBestInliers),
                   curInliersInds = inlierVectorsPool.get(extractCurInliers);
  bestInliersInds.resize(0);
  curInliersInds.resize(0);

  if (doLogFrontPoints)
    VLOG(1) << "points in front: ";

  for (SE3 sol : solutions) {
    curInliersInds = inliersInds;
    int curFrontPointsNum = findInliersMotion(sol, curInliersInds);

    if (doLogFrontPoints)
      VLOG(1) << curFrontPointsNum << ' ';

    if (curFrontPointsNum > bestFrontPointsNum) {
      bestFrontPointsNum = curFrontPointsNum;
      bestSol = sol;
      std::swap(curInliersInds, bestInliersInds);
    }
  }

  if (doLogFrontPoints)
    VLOG(1) << std::endl;

  std::swap(inliersInds, bestInliersInds);
  newInliers = bestFrontPointsNum;
  return bestSol;
}

SE3 StereoGeometryEstimator::findCoarseMotion() {
  if (coarseFound || preciseFound)
    return motion;

  relative_pose::GeneralizedCentralRelativePoseEstimator<double> est;
  constexpr int N = settingEssentialMinimalSolveN;
  const double p = settingEssentialSuccessProb;

  SE3 bestMotion;

  std::mt19937 mt;
  std::uniform_int_distribution<> inds(0, rays.size() - 1);

  int hypotesisInd[N];
  std::pair<Vec3 *, Vec3 *> hypotesis[N];
  Mat33 results[10];

  int bestInliers = -1;
  std::vector<int> curInliersInds = inlierVectorsPool.get(ransacCurInliers),
                   bestInliersInds = inlierVectorsPool.get(ransacBestInliers);

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

    for (int i = 0; i < foundN; ++i) {
      int inliers = findInliersEssential(results[i], curInliersInds);
      if (inliers > maxInliers) {
        maxInliers = inliers;
        maxInliersInd = i;
        std::swap(bestInliersInds, curInliersInds);
      }
    }
    SE3 curMotion = extractMotion(results[maxInliersInd], bestInliersInds,
                                  maxInliers, false);
    if (bestInliers < maxInliers) {
      bestInliers = maxInliers;
      bestMotion = curMotion;
      std::swap(_inliersInds, bestInliersInds);
    }

    double curQ = double(maxInliers) / rays.size();
    if (curQ > q) {
      q = curQ;
      double newIterNum = std::log(1 - p) / std::log(1 - std::pow(curQ, N));
      if (!FLAGS_run_max_RANSAC_iterations)
        iterNum = static_cast<long long>(newIterNum);
    }
  }

  LOG(INFO) << "iterNum = " << iterNum << std::endl;
  LOG(INFO) << "total inliers on coarse = " << bestInliers << std::endl;
  if (iterNum == settingRansacMaxIter)
    LOG(WARNING) << "max number of RANSAC iterations reached" << std::endl;
  coarseFound = true;
  motion = bestMotion;

  if (FLAGS_output_reproj_CDF) {
    std::vector<double> values = reservedVector<double>(rays.size());
    Mat33 E = toEssential(motion);
    Mat33 Et = E.transpose();
    for (int i = 0; i < rays.size(); ++i)
      values.push_back(reprojectionError(cam, E, Et, imgCorresps[i], rays[i]));
    std::sort(values.begin(), values.end());

    std::ofstream ofs(FLAGS_output_directory + "/reproj_err.txt");
    ofs << values.size() << std::endl;
    for (double v : values)
      ofs << v << ' ';
    ofs.close();
  }

  bestMotion =
      extractMotion(toEssential(bestMotion), _inliersInds, bestInliers, true);
  VLOG(1) << "total inliers on coarse after front check = " << bestInliers;

  return bestMotion;
}

struct ReprojectionResidual {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ReprojectionResidual(CameraModel *cam, const Vec3 &p, const Vec3 &q,
                       const Vec2 &pMapped, const Vec2 &qMapped)
      : cam(cam), p(p), q(q), pMapped(pMapped), qMapped(qMapped) {}

  template <typename T>
  bool operator()(const T *const rotP, const T *const transP, T *res) const {
    typedef Eigen::Matrix<T, 2, 1> Vec2t;
    typedef Eigen::Matrix<T, 3, 1> Vec3t;
    typedef Eigen::Matrix<T, 3, 3> Mat33t;
    typedef Eigen::Quaternion<T> Quatt;
    typedef Sophus::SO3<T> SO3t;

    Eigen::Map<const Vec3t> transM(transP);
    Vec3t trans(transM);
    Eigen::Map<const Quatt> rotM(rotP);
    Quatt rot(rotM);

    Vec3t pT = p.cast<T>(), qT = q.cast<T>();
    Vec2t pMappedT = pMapped.cast<T>(), qMappedT = qMapped.cast<T>();

    Mat33t E = SO3t::hat(trans) * rot.toRotationMatrix();
    Vec3t Ep = E * pT, Etq = E.transpose() * qT;

    T EpSqN = Ep.squaredNorm(), EtqSqN = Etq.squaredNorm();
    if (EpSqN < T(1e-4) || EtqSqN < T(1e-4)) {
      res[0] = T(0.0);
      res[1] = T(0.0);
      res[2] = T(0.0);
      res[3] = T(0.0);
      return true;
    }

    Vec3t pProj = pT - (pT.dot(Etq) / EtqSqN) * Etq;
    Vec3t qProj = qT - (qT.dot(Ep) / EpSqN) * Ep;

    Vec2t errVec1 = cam->map(pProj.data()) - pMappedT;
    Vec2t errVec2 = cam->map(qProj.data()) - qMappedT;

    res[0] = errVec1[0];
    res[1] = errVec1[1];
    res[2] = errVec2[0];
    res[3] = errVec2[1];

    return true;
  }

private:
  CameraModel *cam;
  Vec3 p, q;
  Vec2 pMapped, qMapped;
};

SE3 StereoGeometryEstimator::findPreciseMotion() {
  if (preciseFound)
    return motion;
  if (!coarseFound)
    findCoarseMotion();

  SE3 coarseMotion = motion;

  //  std::vector<int> inlierInds;
  //  inlierInds.reserve(inliersNum);
  //  std::cout << "inliers before cleanup = " << inliersNum << std::endl;
  //  for (int i = 0; i < inliersMask.size(); ++i)
  //    if (inliersMask[i])
  //      inlierInds.push_back(i);
  //  std::cout << "inliers in inds = " << inlierInds.size() << std::endl;

  //  std::sort(inlierInds.begin(), inlierInds.end(), [&](int i1, int i2) {
  //    ReprojectionResidual res1(cam, raysNorm[i1].first,
  //    raysNorm[i1].second,
  //                              imgCorresps[i1].first,
  //                              imgCorresps[i1].second);
  //    ReprojectionResidual res2(cam, raysNorm[i2].first,
  //    raysNorm[i2].second,
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

  //  std::cout << inlierInds.size() << ' ' << settingRemoveResidualsRatio <<
  //  '
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

  // Scale of the dso is chosen here
  problem.AddParameterBlock(
      motion.translation().data(), 3,
      new ceres::AutoDiffLocalParameterization<SphericalPlus, 3, 2>(
          new SphericalPlus(Vec3::Zero(), 1, motion.translation())));

  for (int i : _inliersInds)
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<ReprojectionResidual, 4, 4, 3>(
            new ReprojectionResidual(cam, rays[i].first, rays[i].second,
                                     imgCorresps[i].first,
                                     imgCorresps[i].second)),
        new ceres::HuberLoss(std::sqrt(2.0) *
                             settingEssentialReprojErrThreshold),
        motion.so3().data(), motion.translation().data());

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  // options.minimizer_progress_to_stdout = true;
  options.max_num_iterations = 10;
  ceres::Solver::Summary summary;

  ceres::Solve(options, &problem, &summary);

  LOG(INFO) << "post-RANSAC averaging:\n" << summary.FullReport() << std::endl;

  // std::cout << summary.FullReport() << "\n";
  //  std::cout << "avg residual before = "
  //            << std::sqrt(summary.initial_cost / summary.num_residuals)
  //            << "\navg residual after = "
  //            << std::sqrt(summary.final_cost / summary.num_residuals)
  //            << std::endl;

  findInliersEssential(toEssential(motion), _inliersInds);
  findInliersMotion(motion, _inliersInds);

  LOG(INFO) << "translation diff angle = "
            << 180. / M_PI *
                   (coarseMotion.translation() - motion.translation()).norm()
            << "\nrotation diff = "
            << 180. / M_PI *
                   (coarseMotion.so3() * motion.so3().inverse()).log().norm()
            << std::endl;

  preciseFound = true;
  return motion;
}

} // namespace fishdso
