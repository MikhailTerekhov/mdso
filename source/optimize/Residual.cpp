#include "optimize/Residual.h"
#include "internal/system/PreKeyFrameEntryInternals.h"

namespace mdso::optimize {

Vec3t remapDepthed(const SE3t &frameToFrame, const Vec3t &ray, T depth,
                   T maxDepth) {
  if (depth < maxDepth)
    return frameToFrame * (depth * ray);
  else
    return frameToFrame.so3() * ray;
}

Residual::Residual(int hostInd, int hostCamInd, int targetInd, int targetCamInd,
                   int pointInd, CameraBundle *cam, KeyFrameEntry *hostFrame,
                   KeyFrameEntry *targetFrame, OptimizedPoint *optimizedPoint,
                   T logDepth, const SE3t &hostToTargetImage,
                   ceres::LossFunction *lossFunction,
                   const ResidualSettings &settings)
    : mHostInd(hostInd)
    , mHostCamInd(hostCamInd)
    , mTargetInd(targetInd)
    , mTargetCamInd(targetCamInd)
    , mPointInd(pointInd)
    , lossFunction(lossFunction)
    , camTarget(&cam->bundle[targetCamInd].cam)
    , target(targetFrame)
    , hostPoint(optimizedPoint->p.cast<T>())
    , hostDir(optimizedPoint->dir.cast<T>())
    , settings(settings)
    , reprojPattern(settings.residualPattern.pattern().size())
    , hostIntensities(settings.residualPattern.pattern().size())
    , gradWeights(settings.residualPattern.pattern().size()) {
  CameraModel *camHost = &cam->bundle[hostCamInd].cam;
  T depth = exp(logDepth);
  Vec2t reproj = camTarget->map(
      remapDepthed(hostToTargetImage, hostDir, depth, settings.depth.max));
  for (int i = 0; i < reprojPattern.size(); ++i) {
    Vec2t r = camTarget->map(
        hostToTargetImage *
        (depth * camHost
                     ->unmap((hostPoint +
                              settings.residualPattern.pattern()[i].cast<T>())
                                 .eval())
                     .normalized()));
    reprojPattern[i] = r - reproj;
  }

  PreKeyFrameEntryInternals::Interpolator_t *hostInterp =
      &hostFrame->preKeyFrameEntry->internals->interpolator(0);
  for (int i = 0; i < hostIntensities.size(); ++i) {
    Vec2 p = optimizedPoint->p + settings.residualPattern.pattern()[i];
    double hostIntensity;
    hostInterp->Evaluate(p[1], p[0], &hostIntensity);
    hostIntensities[i] = T(hostIntensity);
    double gradNorm = hostFrame->preKeyFrameEntry->gradNorm(toCvPoint(p));
    if (settings.residualWeighting.useGradientWeighting) {
      T c = T(settings.residualWeighting.c);
      gradWeights[i] = c / std::sqrt(c * c + gradNorm * gradNorm);
    } else {
      gradWeights[i] = 1;
    }
  }
}

Residual::Jacobian::DiffFrameParams::DiffFrameParams(int patternSize)
    : dp_dq(Mat24t::Zero())
    , dp_dt(Mat23t::Zero())
    , dr_dab(MatR2t::Zero(patternSize, 2)) {}

Residual::Jacobian::Jacobian(int patternSize)
    : dhost(patternSize)
    , dtarget(patternSize)
    , dp_dlogd(Vec2t::Zero())
    , gradItarget(MatR2t::Zero(patternSize, 2))
    , isInfDepth(false)
    , patternSize(patternSize) {}

Residual::FrameFrameHessian::FrameFrameHessian()
    : qtqt(Mat77t::Zero())
    , qtab(Mat72t::Zero())
    , abqt(Mat27t::Zero())
    , abab(Mat22t::Zero()) {}

Residual::FrameFrameHessian &Residual::FrameFrameHessian::
operator+=(const Residual::FrameFrameHessian &other) {
  qtqt += other.qtqt;
  qtab += other.qtab;
  abqt += other.abqt;
  abab += other.abab;
  return *this;
}

Residual::FrameFrameHessian Residual::FrameFrameHessian::transpose() const {
  FrameFrameHessian transposed;
  transposed.qtqt = qtqt.transpose();
  transposed.qtab = abqt.transpose();
  transposed.abqt = qtab.transpose();
  transposed.abab = abab.transpose();
  return transposed;
}

Residual::FramePointHessian::FramePointHessian()
    : qtd(Vec7t::Zero())
    , abd(Vec2t::Zero()) {}

Residual::FramePointHessian &Residual::FramePointHessian::
operator+=(const Residual::FramePointHessian &other) {
  qtd += other.qtd;
  abd += other.abd;
  return *this;
}

Residual::DeltaHessian::DeltaHessian()
    : pointPoint(0) {}

Residual::FrameGradient::FrameGradient()
    : qt(Vec7t::Zero())
    , ab(Vec2t::Zero()) {}

Residual::FrameGradient &Residual::FrameGradient::
operator+=(const FrameGradient &other) {
  qt += other.qt;
  ab += other.ab;
  return *this;
}

Residual::DeltaGradient::DeltaGradient()
    : point(0) {}

Residual::CachedValues::CachedValues(int patternSize)
    : reproj(Vec2::Zero())
    , gradItarget(MatR2t::Zero(patternSize, 2))
    , depth(0)
    , lightHostToTargetExpA(0) {}

VecRt Residual::getValues(const SE3t &hostToTargetImage,
                          const AffLightT &lightHostToTarget, T logDepth,
                          CachedValues *cachedValues) const {
  auto &targetInterp = target->preKeyFrameEntry->internals->interpolator(0);
  VecRt result(settings.residualPattern.pattern().size());
  T depth = exp(logDepth);
  double lightHostToTargetExpA = lightHostToTarget.ea();
  Vec2t reproj = camTarget->map(
      remapDepthed(hostToTargetImage, hostDir, depth, settings.depth.max));
  if (cachedValues) {
    cachedValues->depth = depth;
    cachedValues->reproj = reproj.cast<double>();
    cachedValues->lightHostToTargetExpA = lightHostToTargetExpA;
  }
  for (int i = 0; i < result.size(); ++i) {
    Vec2t p = reproj + reprojPattern[i];
    double targetIntensity = INF;
    if (cachedValues) {
      Vec2 gradItarget;
      targetInterp.Evaluate(p[1], p[0], &targetIntensity, &gradItarget[1],
                            &gradItarget[0]);
      cachedValues->gradItarget.row(i) = gradItarget.transpose().cast<T>();
    } else
      targetInterp.Evaluate(p[1], p[0], &targetIntensity);
    T hostIntensity =
        lightHostToTarget(hostIntensities[i], lightHostToTargetExpA);
    result[i] = T(targetIntensity) - hostIntensity;
  }

  return result;
}

double Residual::getDeltaEnergy(const VecRt &values) const {
  CHECK_EQ(values.size(), reprojPattern.size());
  double deltaEnergy = 0;
  for (int i = 0; i < values.size(); ++i) {
    double v = values[i];
    double v2 = v * v;
    double rho[3];
    lossFunction->Evaluate(v2, rho);
    deltaEnergy += gradWeights[i] * rho[0];
  }
  return deltaEnergy;
}

VecRt Residual::getHessianWeights(const VecRt &values) const {
  VecRt weights(settings.residualPattern.pattern().size());
  for (int i = 0; i < weights.size(); ++i) {
    double v = values[i];
    double v2 = v * v;
    double rho[3];
    lossFunction->Evaluate(v2, rho);
    double w = rho[1] + 2 * rho[2] * v2;
    if (w < 0) {
      CHECK_GE(rho[1], 0);
      w = settings.residualWeighting.lossEps * rho[1];
    }
    weights[i] = gradWeights[i] * w;
  }
  return weights;
}

VecRt Residual::getGradientWeights(const VecRt &values) const {
  VecRt weights(settings.residualPattern.pattern().size());
  for (int i = 0; i < weights.size(); ++i) {
    double v = values[i];
    double v2 = v * v;
    double rho[3];
    lossFunction->Evaluate(v2, rho);
    weights[i] = gradWeights[i] * rho[1];
  }
  return weights;
}

VecRt Residual::getPixelDependentWeights() const { return gradWeights; }

Residual::Jacobian Residual::getJacobian(
    const SE3t &hostToTarget, const MotionDerivatives &dHostToTarget,
    const AffLightT &lightWorldToHost, const AffLightT &lightHostToTarget,
    T logDepth, const CachedValues &cachedValues) const {
  Jacobian jacobian(settings.residualPattern.pattern().size());
  PreKeyFrameEntryInternals::Interpolator_t *targetInterp =
      &target->preKeyFrameEntry->internals->interpolator(0);

  T depth = cachedValues.depth;

  jacobian.isInfDepth = false;
  if (depth > settings.depth.max) {
    jacobian.isInfDepth = true;
    depth = settings.depth.max;
  }
  Vec3t hostVec = depth * hostDir;
  Vec4t hostVecH = makeHomogeneous(hostVec);
  Vec3t targetVec = hostToTarget * hostVec;

  auto [reproj, dpi] = camTarget->diffMap(targetVec);

  jacobian.gradItarget = cachedValues.gradItarget;

  jacobian.dp_dlogd = dpi * (hostToTarget.so3() * hostVec);
  jacobian.dhost.dp_dq = dpi * dHostToTarget.daction_dq_host(hostVecH);
  jacobian.dhost.dp_dt = dpi * dHostToTarget.daction_dt_host;
  jacobian.dtarget.dp_dq = dpi * dHostToTarget.daction_dq_target(hostVecH);
  jacobian.dtarget.dp_dt = dpi * dHostToTarget.daction_dt_target;

  T lightHostToTargetExpA = cachedValues.lightHostToTargetExpA;

  for (int i = 0; i < settings.residualPattern.pattern().size(); ++i) {
    double d_da =
        lightHostToTargetExpA * (hostIntensities[i] - lightWorldToHost.b());
    jacobian.dhost.dr_dab(i, 0) = d_da;
    jacobian.dhost.dr_dab(i, 1) = lightHostToTargetExpA;
    jacobian.dtarget.dr_dab(i, 0) = -d_da;
    jacobian.dtarget.dr_dab(i, 1) = -1;
  }

  return jacobian;
}

inline Mat22t sum_gradab(const VecRt &weights, const MatR2t gradItarget,
                         const MatR2t dr_dab) {
  return gradItarget.transpose() * weights.asDiagonal() * dr_dab;
}

template <bool isSameFrame>
inline Residual::FrameFrameHessian
H_frameframe(const Mat27t &df1_dp_dqt, const Mat27t &df2_dp_dqt,
             const MatR2t df1_dr_dab, const MatR2t df2_dr_dab,
             const VecRt &weights, const Mat22t &sum_wgradgradT,
             const Mat22t &sum_gradab1, const Mat22t &sum_gradab2) {
  Residual::FrameFrameHessian H;
  H.qtqt = df1_dp_dqt.transpose() * sum_wgradgradT * df2_dp_dqt;
  H.qtab = df1_dp_dqt.transpose() * sum_gradab2;
  if constexpr (isSameFrame)
    H.abqt = H.qtab.transpose();
  else
    H.abqt = sum_gradab1.transpose() * df2_dp_dqt;
  H.abab = df1_dr_dab.transpose() * weights.asDiagonal() * df2_dr_dab;
  return H;
}

inline Residual::FramePointHessian H_framepoint(const Mat27t &dp_dqt,
                                                const Vec2t &dp_dlogd,
                                                const Mat22t &sum_wgradgradT,
                                                const Mat22t &sum_gradab) {
  Residual::FramePointHessian H;
  H.abd = sum_gradab.transpose() * dp_dlogd;
  H.qtd = dp_dqt.transpose() * sum_wgradgradT * dp_dlogd;
  return H;
}

Residual::DeltaHessian
Residual::getDeltaHessian(const VecRt &values,
                          const Residual::Jacobian &jacobian) const {
  VecRt weights = getHessianWeights(values);

  DeltaHessian deltaHessian;

  Mat27t dhost_dp_dqt;
  dhost_dp_dqt << jacobian.dhost.dp_dq, jacobian.dhost.dp_dt;
  Mat27t dtarget_dp_dqt;
  dtarget_dp_dqt << jacobian.dtarget.dp_dq, jacobian.dtarget.dp_dt;

  Mat22t sum_wgradgradT = jacobian.gradItarget.transpose() *
                          weights.asDiagonal() * jacobian.gradItarget;
  Mat22t sum_gradab_host =
      sum_gradab(weights, jacobian.gradItarget, jacobian.dhost.dr_dab);
  Mat22t sum_gradab_target =
      sum_gradab(weights, jacobian.gradItarget, jacobian.dtarget.dr_dab);

  deltaHessian.hostHost = H_frameframe<true>(
      dhost_dp_dqt, dhost_dp_dqt, jacobian.dhost.dr_dab, jacobian.dhost.dr_dab,
      weights, sum_wgradgradT, sum_gradab_host, sum_gradab_host);
  deltaHessian.hostTarget =
      H_frameframe<false>(dhost_dp_dqt, dtarget_dp_dqt, jacobian.dhost.dr_dab,
                          jacobian.dtarget.dr_dab, weights, sum_wgradgradT,
                          sum_gradab_host, sum_gradab_target);
  deltaHessian.targetTarget = H_frameframe<true>(
      dtarget_dp_dqt, dtarget_dp_dqt, jacobian.dtarget.dr_dab,
      jacobian.dtarget.dr_dab, weights, sum_wgradgradT, sum_gradab_target,
      sum_gradab_target);

  deltaHessian.hostPoint = H_framepoint(dhost_dp_dqt, jacobian.dp_dlogd,
                                        sum_wgradgradT, sum_gradab_host);
  deltaHessian.targetPoint = H_framepoint(dtarget_dp_dqt, jacobian.dp_dlogd,
                                          sum_wgradgradT, sum_gradab_target);

  deltaHessian.pointPoint =
      jacobian.dp_dlogd.dot(sum_wgradgradT * jacobian.dp_dlogd);

  return deltaHessian;
}

inline Residual::FrameGradient
getFrameGradient(const Residual::Jacobian::DiffFrameParams &dframe,
                 const Vec2t &gradWR, const VecRt &wr) {
  Residual::FrameGradient frameGradient;
  frameGradient.qt.head<4>() = dframe.dp_dq.transpose() * gradWR;
  frameGradient.qt.tail<3>() = dframe.dp_dt.transpose() * gradWR;
  frameGradient.ab = dframe.dr_dab.transpose() * wr;
  return frameGradient;
}

Residual::DeltaGradient
Residual::getDeltaGradient(const VecRt &values,
                           const Residual::Jacobian &jacobian) const {
  DeltaGradient deltaGradient;

  VecRt weights = getGradientWeights(values);
  VecRt wr = weights.cwiseProduct(values);
  Vec2t gradWR = jacobian.gradItarget.transpose() * wr;
  deltaGradient.host = getFrameGradient(jacobian.dhost, gradWR, wr);
  deltaGradient.target = getFrameGradient(jacobian.dtarget, gradWR, wr);
  deltaGradient.point = jacobian.dp_dlogd.dot(gradWR);
  return deltaGradient;
}

std::ostream &operator<<(std::ostream &os, const Residual &res) {
  os << "host ind = " << res.hostInd()
     << "\nhost cam ind = " << res.hostCamInd()
     << "\ntarget ind = " << res.targetInd()
     << "\ntarget cam ind = " << res.targetCamInd()
     << "\npoint ind = " << res.pointInd()
     << "\nhost point = " << res.hostPoint.transpose()
     << "\nhost dir = " << res.hostDir.transpose() << "\n";
  return os;
}

MatR4t Residual::Jacobian::dr_dq_host() const {
  return gradItarget * dhost.dp_dq;
}

MatR3t Residual::Jacobian::dr_dt_host() const {
  return gradItarget * dhost.dp_dt;
}

MatR4t Residual::Jacobian::dr_dq_target() const {
  return gradItarget * dtarget.dp_dq;
}

MatR3t Residual::Jacobian::dr_dt_target() const {
  return gradItarget * dtarget.dp_dt;
}

MatR2t Residual::Jacobian::dr_daff_host() const { return dhost.dr_dab; }

MatR2t Residual::Jacobian::dr_daff_target() const { return dtarget.dr_dab; }

VecRt Residual::Jacobian::dr_dlogd() const { return gradItarget * dp_dlogd; }

MatRx19t Residual::Jacobian::dr_dparams() const {
  MatRx19t result(patternSize, 19);
  result << dr_dq_host(), dr_dt_host(), dr_dq_target(), dr_dt_target(),
      dr_daff_host(), dr_daff_target(), dr_dlogd();
  return result;
}

} // namespace mdso::optimize
