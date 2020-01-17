#ifndef INCLUDE_RESIDUAL
#define INCLUDE_RESIDUAL

#include "optimize/MotionDerivatives.h"
#include "system/CameraBundle.h"
#include "system/KeyFrame.h"
#include <ceres/loss_function.h>

namespace mdso::optimize {

using VecRt = Eigen::Matrix<T, Eigen::Dynamic, 1, Eigen::ColMajor,
                            Settings::ResidualPattern::max_size>;
using MatR2t = Eigen::Matrix<T, Eigen::Dynamic, 2, Eigen::ColMajor,
                             Settings::ResidualPattern::max_size>;
using MatR3t = Eigen::Matrix<T, Eigen::Dynamic, 3, Eigen::ColMajor,
                             Settings::ResidualPattern::max_size>;
using MatR4t = Eigen::Matrix<T, Eigen::Dynamic, 4, Eigen::ColMajor,
                             Settings::ResidualPattern::max_size>;
using MatR5t = Eigen::Matrix<T, Eigen::Dynamic, 5, Eigen::ColMajor,
                             Settings::ResidualPattern::max_size>;
using MatR6t = Eigen::Matrix<T, Eigen::Dynamic, 6, Eigen::ColMajor,
                             Settings::ResidualPattern::max_size>;
using MatR7t = Eigen::Matrix<T, Eigen::Dynamic, 7, Eigen::ColMajor,
                             Settings::ResidualPattern::max_size>;

class Residual {
public:
  static constexpr int MPS = Settings::ResidualPattern::max_size;

  Residual(int hostInd, int hostCamInd, int targetInd, int targetCamInd,
           int pointInd, const T *logDepth, CameraBundle *cam,
           KeyFrameEntry *hostFrame, KeyFrameEntry *targetFrame,
           OptimizedPoint *optimizedPoint, const SE3t &hostToTargetImage,
           ceres::LossFunction *lossFunction, const ResidualSettings &settings);

  struct FrameFrameHessian {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    FrameFrameHessian();
    FrameFrameHessian &operator+=(const FrameFrameHessian &other);
    FrameFrameHessian transpose() const;

    Eigen::Matrix<T, SE3t::num_parameters, SE3t::num_parameters> qtqt;
    Eigen::Matrix<T, SE3t::num_parameters, AffLightT::DoF> qtab;
    Eigen::Matrix<T, AffLightT::DoF, SE3t::num_parameters> abqt;
    Eigen::Matrix<T, AffLightT::DoF, AffLightT::DoF> abab;
  };

  struct FramePointHessian {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    FramePointHessian();
    FramePointHessian &operator+=(const FramePointHessian &other);

    Eigen::Matrix<T, SE3t::num_parameters, 1> qtd;
    Eigen::Matrix<T, AffLightT::DoF, 1> abd;
  };

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

    MatR4t dr_dq_host(int patternSize) const;
    MatR3t dr_dt_host(int patternSize) const;
    MatR4t dr_dq_target(int patternSize) const;
    MatR3t dr_dt_target(int patternSize) const;
    MatR2t dr_daff_host(int patternSize) const;
    MatR2t dr_daff_target(int patternSize) const;
    VecRt dr_dlogd(int patternSize) const;
  };

  struct DeltaHessian {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    DeltaHessian();

    FrameFrameHessian hostHost;
    FrameFrameHessian hostTarget;
    FrameFrameHessian targetTarget;

    FramePointHessian hostPoint;
    FramePointHessian targetPoint;

    T pointPoint;
  };

  inline int hostInd() const { return mHostInd; }
  inline int hostCamInd() const { return mHostCamInd; }
  inline int targetInd() const { return mTargetInd; }
  inline int targetCamInd() const { return mTargetCamInd; }
  inline int pointInd() const { return mPointInd; }

  inline static_vector<Vec2, MPS> getReprojPattern() const {
    return reprojPattern;
  }
  inline static_vector<double, MPS> getHostIntensities() const {
    return hostIntensities;
  }

  inline static_vector<T, MPS>
  getValues(const SE3t &hostToTargetImage,
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

  static DeltaHessian getDeltaHessian(const static_vector<T, MPS> &values,
                                      const static_vector<T, MPS> &weights,
                                      const Residual::Jacobian &jacobian);

  friend std::ostream &operator<<(std::ostream &os, const Residual &res);

private:
  int mHostInd;
  int mHostCamInd;
  int mTargetInd;
  int mTargetCamInd;
  int mPointInd;

  const T *logDepth;

  ceres::LossFunction *lossFunction;
  CameraModel *camTarget;
  KeyFrameEntry *target;
  Vec2t hostPoint;
  Vec3t hostDir;
  const ResidualSettings &settings;
  static_vector<Vec2t, MPS> reprojPattern;
  static_vector<T, MPS> hostIntensities;
  static_vector<T, MPS> gradWeights;
};

} // namespace mdso::optimize

#endif
