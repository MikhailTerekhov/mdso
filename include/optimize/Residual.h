#ifndef INCLUDE_RESIDUAL
#define INCLUDE_RESIDUAL

#include "system/CameraBundle.h"
#include "system/KeyFrame.h"
#include "optimize/MotionDerivatives.h"
#include <ceres/loss_function.h>

namespace mdso::optimize {

class Residual {
public:
  static constexpr int MPS = Settings::ResidualPattern::max_size;

  Residual(CameraBundle::CameraEntry *camHost,
           CameraBundle::CameraEntry *camTarget, KeyFrameEntry *host,
           KeyFrameEntry *target, OptimizedPoint *optimizedPoint,
           const SE3 &hostToTarget, ceres::LossFunction *lossFunction,
           const ResidualSettings &settings);

  struct Jacobian {
    static constexpr int MPS = Settings::ResidualPattern::max_size;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    struct DiffFrameParams {
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW

      Eigen::Matrix<T, 2, SO3t::num_parameters> dp_dq;
      Vec2t dr_dab[MPS];
    };

    Mat23t dpi;

    DiffFrameParams dhost;
    DiffFrameParams dtarget;
    Vec2t dp_dlogd;

    Vec2t gradItarget[MPS];
  };

  struct DeltaHessian {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    struct FrameFrame {
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW

      Eigen::Matrix<T, SE3t::num_parameters, SE3t::num_parameters> qtqt;
      Eigen::Matrix<T, SE3t::num_parameters, AffLightT::DoF> qtab;
      Eigen::Matrix<T, AffLightT::DoF, AffLightT::DoF> abab;
    };

    struct FramePoint {
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW

      Eigen::Matrix<T, SE3t::num_parameters, 1> qtd;
      Eigen::Matrix<T, AffLightT::DoF, 1> abd;
    };

    FrameFrame hostHost;
    FrameFrame hostTarget;
    FrameFrame targetTarget;

    FramePoint hostPoint;
    FramePoint targetPoint;

    T pointPoint;
  };

  static_vector<T, MPS> getValues(const SE3t &hostToTarget,
                                  const AffLightT &lightHostToTarget);
  static_vector<T, MPS> getWeights(const static_vector<T, MPS> &values);
  Jacobian getJacobian(const SE3t &hostToTarget,
                       const MotionDerivatives &dHostToTarget,
                       const AffLightT &lightHostToTarget,
                       const Mat33t &worldToTargetRot);
  DeltaHessian getDeltaHessian(const Residual::Jacobian &jacobian,
                               const MotionDerivatives &dHostToTarget,
                               const SE3t &hostToTarget,
                               const AffLightT &lightWorldToTarget);

private:
  ceres::LossFunction *lossFunction;
  CameraBundle::CameraEntry *camHost;
  CameraBundle::CameraEntry *camTarget;
  KeyFrameEntry *host;
  KeyFrameEntry *target;
  OptimizedPoint *optimizedPoint;
  const ResidualSettings &settings;
  static_vector<Vec2, MPS> reprojPattern;
  static_vector<double, MPS> hostIntencities;
  static_vector<double, MPS> gradWeights;
};

} // namespace mdso::optimize

#endif
