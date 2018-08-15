/*
 * Closed-form 5-point relative pose solver
 *
 * - Groebner-bases based solution
 * - Solution extraction using eigenvalues
 */
#ifndef GENERALIZED_RELATIVE_POSE
#define GENERALIZED_RELATIVE_POSE

#include <Eigen/Eigen>

namespace relative_pose {
template <typename Scalar, int npoints = 5, bool values = true>

struct GeneralizedCentralRelativePoseEstimator {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  typedef Eigen::Matrix<Scalar, 10, 1> Vector10;
  typedef Eigen::Matrix<Scalar, 9, 1> Vector9;
  typedef Eigen::Matrix<Scalar, 3, 1> Vector3;
  typedef Eigen::Matrix<Scalar, 3, 3> Matrix33;
  typedef Eigen::Matrix<Scalar, 3, 10> Matrix310;
  typedef Eigen::Matrix<Scalar, 10, 10> Matrix1010;
  typedef Eigen::Matrix<Scalar, 9, 9> Matrix99;

  using Projection = Vector3;
  const static int NPOINTS = npoints;
  const static int NSOLS = 10;

  GeneralizedCentralRelativePoseEstimator() {}
  int estimate(std::pair<Projection *, Projection *> *p, Matrix33 *res) {
    XLSE = Eigen::Matrix<Scalar, 9, NPOINTS>();
    return estimateImpl(p, NPOINTS, res);
  }

  int estimate(std::pair<Vector3 *, Vector3 *> *rays, int N, Matrix33 *res) {
    XLSE = Eigen::Matrix<Scalar, 9, NPOINTS>(9, N);
    return estimateImpl(rays, N, res);
  }

  int estimate(const Projection **l, const Projection **r, Matrix33 *res) {
    for (int i = 0; i < NPOINTS; ++i) {
      Matrix33 prod = *l[i] * r[i]->transpose();
      int idx = 0;
      for (int ii = 0; ii < 3; ++ii)
        for (int jj = 0; jj < 3; ++jj)
          XLSE(idx++, i) = prod(jj, ii);
    }
    return solve(res);
  }

private:
  int estimateImpl(std::pair<Projection *, Projection *> *p, int N,
                   Matrix33 *res) {
    for (int i = 0; i < N; ++i) {
      auto l = p[i].first, r = p[i].second;
      Matrix33 prod = *l * r->transpose();
      int idx = 0;
      for (int ii = 0; ii < 3; ++ii)
        for (int jj = 0; jj < 3; ++jj)
          XLSE(idx++, i) = prod(jj, ii);
    }
    return solve(res);
  }
  int solve(Matrix33 *res);

  Eigen::Matrix<Scalar, 9, NPOINTS> XLSE;
  Eigen::Matrix<Scalar, 10, 20> T;
  Matrix1010 M;
};

extern template struct GeneralizedCentralRelativePoseEstimator<double>;

} // namespace relative_pose

#endif
