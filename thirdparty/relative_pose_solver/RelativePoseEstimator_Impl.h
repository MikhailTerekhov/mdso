#ifndef GENERALIZED_RELATIVE_POSE_IMPL
#define GENERALIZED_RELATIVE_POSE_IMPL

namespace relative_pose {

template <typename Scalar, int npoints, bool values>
int GeneralizedCentralRelativePoseEstimator<Scalar, npoints, values>::solve(
    Matrix33 *res) {
  Eigen::FullPivHouseholderQR<decltype(XLSE)> xqr(XLSE);
  Matrix99 Q = xqr.matrixQ();
  Vector9 X = Q.col(5), Y = Q.col(6), Z = Q.col(7), W = Q.col(8);

  T.setZero();
  Matrix33 X_, Y_, Z_, W_;
  int idx = 0;
  for (int ii = 0; ii < 3; ++ii) {
    for (int jj = 0; jj < 3; ++jj) {
      X_(jj, ii) = X[idx];
      Y_(jj, ii) = Y[idx];
      Z_(jj, ii) = Z[idx];
      W_(jj, ii) = W[idx];
      ++idx;
    }
  }

#include "gcrp.h"
  Matrix1010 rhs = T.template block<10, 10>(0, 10);
  Matrix1010 lhs = T.template block<10, 10>(0, 0);
  Matrix1010 B = lhs.fullPivHouseholderQr().solve(rhs);

  Matrix1010 Ax, Ay, Az;
  Ax.template block<4, 10>(6, 0).setZero();
  Ax.row(0) = -B.row(0);
  Ax.row(1) = -B.row(1);
  Ax.row(2) = -B.row(2);
  Ax.row(3) = -B.row(4);
  Ax.row(4) = -B.row(5);
  Ax.row(5) = -B.row(7);
  Ax(6, 0) = Scalar(1);
  Ax(7, 1) = Scalar(1);
  Ax(8, 3) = Scalar(1);
  Ax(9, 6) = Scalar(1);

  idx = 0;
  if (!values) {
    Eigen::EigenSolver<Matrix1010> eig(Ax);
    Eigen::Matrix<std::complex<Scalar>, 3, 10> xyz1 =
        eig.eigenvectors().template block<3, 10>(6, 0);
    Eigen::Matrix<std::complex<Scalar>, 10, 1> scaler =
        eig.eigenvectors().template block<1, 10>(9, 0).transpose().eval();
    Matrix310 xyz = (xyz1 * Eigen::DiagonalMatrix<std::complex<Scalar>, 10>(
                                (Scalar(1) / scaler.array()).matrix()))
                        .real();

    for (int i = 0; i < 10; ++i) {
      if (std::abs(eig.eigenvalues()[i].imag()) * 1e6 <
          std::abs(eig.eigenvalues()[i].real()))
        res[idx++] = xyz(0, i) * X_ + xyz(1, i) * Y_ + xyz(2, i) * Z_ + W_;
    }
    return idx;
  }

  Eigen::EigenSolver<Matrix1010> eig(Ax);

  Vector10 xs = eig.eigenvalues().real();
  Ay.template block<4, 10>(6, 0).setZero();
  Ay.row(0) = -B.row(1);
  Ay.row(1) = -B.row(2);
  Ay.row(2) = -B.row(3);
  Ay.row(3) = -B.row(5);
  Ay.row(4) = -B.row(6);
  Ay.row(5) = -B.row(8);
  Ay(6, 1) = Scalar(1);
  Ay(7, 2) = Scalar(1);
  Ay(8, 4) = Scalar(1);
  Ay(9, 7) = Scalar(1);

  Az.template block<4, 10>(6, 0).setZero();
  Az.row(0) = -B.row(4);
  Az.row(1) = -B.row(5);
  Az.row(2) = -B.row(6);
  Az.row(3) = -B.row(7);
  Az.row(4) = -B.row(8);
  Az.row(5) = -B.row(9);
  Az(6, 3) = Scalar(1);
  Az(7, 4) = Scalar(1);
  Az(8, 5) = Scalar(1);
  Az(9, 8) = Scalar(1);

  typedef Eigen::Matrix<std::complex<Scalar>, 10, 10> EV;
  EV V(eig.eigenvectors());
  Eigen::FullPivHouseholderQR<EV> qr(V);
  Vector10 ys = qr.solve(Ay * V).diagonal().real();
  Vector10 zs = qr.solve(Az * V).diagonal().real();
  for (int i = 0; i < 10; ++i) {
    if (std::abs(eig.eigenvalues()[i].imag()) * 1e6 <
        std::abs(eig.eigenvalues()[i].real()))
      res[idx++] = xs[i] * X_ + ys[i] * Y_ + zs[i] * Z_ + W_;
  }
  return idx;
}
}

#endif
