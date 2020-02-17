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
using MatRx19t = Eigen::Matrix<T, Eigen::Dynamic, 19, Eigen::ColMajor,
                               Settings::ResidualPattern::max_size>;

class Residual {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  static constexpr int MPS = Settings::ResidualPattern::max_size;

  struct Jacobian {
    static constexpr int MPS = Settings::ResidualPattern::max_size;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    struct DiffFrameParams {
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW

      DiffFrameParams(int patternSize);

      Eigen::Matrix<T, 2, SO3t::num_parameters> dp_dq;
      Mat23t dp_dt;
      MatR2t dr_dab;
    };

    struct CachedValues {
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW

      Vec2 reproj;
      MatR2t gradItarget;
      T expDepth;
      T expA;
    };

    Jacobian(int patternSize);

    DiffFrameParams dhost;
    DiffFrameParams dtarget;
    Vec2t dp_dlogd;

    // each row is a transposed gradient
    MatR2t gradItarget;

    bool isInfDepth;

    MatR4t dr_dq_host(int patternSize) const;
    MatR3t dr_dt_host(int patternSize) const;
    MatR4t dr_dq_target(int patternSize) const;
    MatR3t dr_dt_target(int patternSize) const;
    MatR2t dr_daff_host(int patternSize) const;
    MatR2t dr_daff_target(int patternSize) const;
    VecRt dr_dlogd(int patternSize) const;
    MatRx19t dr_dparams(int patternSize) const;
  };

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

  struct FrameGradient {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    FrameGradient();
    FrameGradient &operator+=(const FrameGradient &other);

    Eigen::Matrix<T, SE3t::num_parameters, 1> qt;
    Eigen::Matrix<T, AffLightT::DoF, 1> ab;
  };

  struct DeltaGradient {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    DeltaGradient();

    FrameGradient host;
    FrameGradient target;

    T point;
  };

  struct CachedValues {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    CachedValues(int patternSize);

    Vec2 reproj;
    MatR2t gradItarget;
    T depth;
    T lightHostToTargetExpA;
  };

  Residual(int hostInd, int hostCamInd, int targetInd, int targetCamInd,
           int pointInd, CameraBundle *cam, KeyFrameEntry *hostFrame,
           KeyFrameEntry *targetFrame, OptimizedPoint *optimizedPoint,
           T logDepth, const SE3t &hostToTargetImage,
           ceres::LossFunction *lossFunction, const ResidualSettings &settings);

  inline int hostInd() const { return mHostInd; }
  inline int hostCamInd() const { return mHostCamInd; }
  inline int targetInd() const { return mTargetInd; }
  inline int targetCamInd() const { return mTargetCamInd; }
  inline int pointInd() const { return mPointInd; }
  inline int patternSize() const { return settings.patternSize(); }

  inline static_vector<Vec2t, MPS> getReprojPattern() const {
    return reprojPattern;
  }
  inline VecRt getHostIntensities() const { return hostIntensities; }
  inline Vec3t getHostDir() const { return hostDir; }

  inline VecRt getValues(const SE3t &hostToTargetImage,
                         const AffLightT &lightHostToTarget, T logDepth) const {
    return getValues(hostToTargetImage, lightHostToTarget, logDepth, nullptr);
  }
  VecRt getValues(const SE3t &hostToTargetImage,
                  const AffLightT &lightHostToTarget, T logDepth,
                  CachedValues *cachedValues) const;
  VecRt getHessianWeights(const VecRt &values) const;
  VecRt getGradientWeights(const VecRt &values) const;
  Jacobian getJacobian(const SE3t &hostToTarget,
                       const MotionDerivatives &dHostToTarget,
                       const AffLightT &lightWorldToHost,
                       const AffLightT &lightHostToTarget, T logDepth,
                       const CachedValues &cachedValues) const;
  DeltaHessian getDeltaHessian(const VecRt &values,
                               const Residual::Jacobian &jacobian) const;
  DeltaGradient getDeltaGradient(const VecRt &values,
                                 const Residual::Jacobian &jacobian) const;

  friend std::ostream &operator<<(std::ostream &os, const Residual &res);

private:
  int mHostInd;
  int mHostCamInd;
  int mTargetInd;
  int mTargetCamInd;
  int mPointInd;

  ceres::LossFunction *lossFunction;
  CameraModel *camTarget;
  KeyFrameEntry *target;
  Vec2t hostPoint;
  Vec3t hostDir;
  const ResidualSettings &settings;
  static_vector<Vec2t, MPS> reprojPattern;
  VecRt hostIntensities;
  VecRt gradWeights;
};

} // namespace mdso::optimize

#endif
