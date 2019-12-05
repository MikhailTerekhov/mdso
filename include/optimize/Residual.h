#ifndef INCLUDE_RESIDUAL
#define INCLUDE_RESIDUAL

#include "optimize/MotionDerivatives.h"
#include "system/CameraBundle.h"
#include "system/KeyFrame.h"
#include <ceres/loss_function.h>

namespace mdso::optimize {

class Residual {
public:
  static constexpr int MPS = Settings::ResidualPattern::max_size;

  Residual(CameraBundle::CameraEntry *camHost,
           CameraBundle::CameraEntry *camTarget, KeyFrameEntry *host,
           KeyFrameEntry *targetFrame, OptimizedPoint *optimizedPoint,
           const SE3 &hostToTargetImage, ceres::LossFunction *lossFunction,
           const ResidualSettings &settings);

  struct Jacobian {
    static constexpr int MPS = Settings::ResidualPattern::max_size;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    struct DiffFrameParams {
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW

      Eigen::Matrix<T, 2, SO3t::num_parameters> dp_dq;
      Eigen::Matrix<T, 2, 3> dp_dt;
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

  inline static_vector<Vec2, MPS> getReprojPattern() const {
    return reprojPattern;
  }
  inline static_vector<double, MPS> getHostIntensities() const {
    return hostIntensities;
  }

  inline static_vector<T, MPS>
  getValues(const SE3 &hostToTargetImage,
            const AffLightT &lightHostToTarget) const {
    return getValues(hostToTargetImage, lightHostToTarget, nullptr);
  }
  static_vector<T, MPS> getValues(const SE3 &hostToTargetImage,
                                  const AffLightT &lightHostToTarget,
                                  Vec2 *reprojOut) const;
  static_vector<T, MPS> getWeights(const static_vector<T, MPS> &values) const;
  Jacobian getJacobian(const SE3t &hostToTarget,
                       const MotionDerivatives &dHostToTarget,
                       const AffLightT &lightWorldToHost,
                       const AffLightT &lightHostToTarget) const;
  DeltaHessian getDeltaHessian(const Residual::Jacobian &jacobian,
                               const MotionDerivatives &dHostToTarget,
                               const SE3 &hostToTarget,
                               const AffLightT &lightWorldToTarget) const;

private:
  ceres::LossFunction *lossFunction;
  CameraBundle::CameraEntry *camHost;
  CameraBundle::CameraEntry *camTarget;
  KeyFrameEntry *host;
  KeyFrameEntry *target;
  OptimizedPoint *optimizedPoint;
  KeyFrameEntry *targetFrame;
  const ResidualSettings &settings;
  static_vector<Vec2, MPS> reprojPattern;
  static_vector<double, MPS> hostIntensities;
  static_vector<double, MPS> gradWeights;
};

} // namespace mdso::optimize

#endif
