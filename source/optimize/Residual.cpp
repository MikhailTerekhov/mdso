#include "optimize/Residual.h"
#include "PreKeyFrameEntryInternals.h"

namespace mdso::optimize {

Vec3t remapDepthed(const SE3t &frameToFrame, const Vec3t &ray, T depth) {
  if (std::isfinite(depth))
    return frameToFrame * (depth * ray);
  else
    return frameToFrame.so3() * ray;
}

Residual::Residual(int hostInd, int hostCamInd, int targetInd, int targetCamInd,
                   int pointInd, const T *logDepth, CameraBundle *cam,
                   KeyFrameEntry *hostFrame, KeyFrameEntry *targetFrame,
                   OptimizedPoint *optimizedPoint,
                   const SE3t &hostToTargetImage,
                   ceres::LossFunction *lossFunction,
                   const ResidualSettings &settings)
    : mHostInd(hostInd)
    , mHostCamInd(hostCamInd)
    , mTargetInd(targetInd)
    , mTargetCamInd(targetCamInd)
    , mPointInd(pointInd)
    , logDepth(logDepth)
    , lossFunction(lossFunction)
    , camTarget(&cam->bundle[targetCamInd].cam)
    , target(targetFrame)
    , hostDir(optimizedPoint->dir.cast<T>())
    , hostPoint(optimizedPoint->p.cast<T>())
    , settings(settings)
    , reprojPattern(settings.residualPattern.pattern().size())
    , hostIntensities(settings.residualPattern.pattern().size())
    , gradWeights(settings.residualPattern.pattern().size()) {
  CameraModel *camHost = &cam->bundle[hostCamInd].cam;
  T depth = exp(*logDepth);
  Vec2t reproj =
      camTarget->map(remapDepthed(hostToTargetImage, hostDir, depth));
  for (int i = 0; i < reprojPattern.size(); ++i) {
    Vec2t r = camTarget->map(
        hostToTargetImage *
        (depth *
         camHost
             ->unmap((hostPoint + settings.residualPattern.pattern()[i]).eval())
             .normalized()));
    reprojPattern[i] = r - reproj;
  }

  PreKeyFrameEntryInternals::Interpolator_t *hostInterp =
      &hostFrame->preKeyFrameEntry->internals->interpolator(0);
  for (int i = 0; i < hostIntensities.size(); ++i) {
    Vec2 p = optimizedPoint->p + settings.residualPattern.pattern()[i];
    Vec2 gradIhost;
    double hostIntensity;
    hostInterp->Evaluate(p[1], p[0], &hostIntensity, &gradIhost[1],
                         &gradIhost[0]);
    hostIntensities[i] = T(hostIntensity);
    T normSq = T(gradIhost.squaredNorm());
    T c = T(settings.residualWeighting.c);
    gradWeights[i] = c / std::sqrt(c * c + normSq);
  }
}

static_vector<T, Residual::MPS>
Residual::getValues(const SE3t &hostToTargetImage,
                    const AffLightT &lightHostToTarget, Vec2 *reprojOut) const {
  auto &targetInterp = target->preKeyFrameEntry->internals->interpolator(0);
  static_vector<T, Residual::MPS> result(
      settings.residualPattern.pattern().size());
  T depth = exp(*logDepth);
  Vec2t reproj =
      camTarget->map(remapDepthed(hostToTargetImage, hostDir, depth));
  for (int i = 0; i < result.size(); ++i) {
    Vec2t p = reproj + reprojPattern[i];
    double targetIntensity = INF;
    targetInterp.Evaluate(p[1], p[0], &targetIntensity);
    T hostIntensity = lightHostToTarget(hostIntensities[i]);
    result[i] = T(targetIntensity) - hostIntensity;
  }

  if (reprojOut)
    *reprojOut = reproj.cast<double>();

  return result;
}

static_vector<T, Residual::MPS>
Residual::getWeights(const static_vector<T, MPS> &values) const {
  static_vector<T, MPS> weights(settings.residualPattern.pattern().size());
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

Residual::Jacobian
Residual::getJacobian(const SE3t &hostToTarget,
                      const MotionDerivatives &dHostToTarget,
                      const AffLightT &lightWorldToHost,
                      const AffLightT &lightHostToTarget) const {
  Jacobian jacobian;
  PreKeyFrameEntryInternals::Interpolator_t *targetInterp =
      &target->preKeyFrameEntry->internals->interpolator(0);

  T depth = exp(*logDepth);
  if (!std::isfinite(depth))
    depth = settings.depth.max;
  Vec3t hostVec = depth * hostDir;
  Vec4t hostVecH = makeHomogeneous(hostVec);
  Vec3t targetVec = hostToTarget * hostVec;
  Vec2t reproj;
  std::tie(reproj, jacobian.dpi) = camTarget->diffMap(targetVec);

  for (int i = 0; i < settings.residualPattern.pattern().size(); ++i) {
    Vec2t p = reproj + reprojPattern[i];
    Vec2 gradItarget;
    double intensity = INF;
    targetInterp->Evaluate(p[1], p[0], &intensity, &gradItarget[1],
                           &gradItarget[0]);
    jacobian.gradItarget[i] = gradItarget.cast<T>();
  }

  jacobian.dp_dlogd = jacobian.dpi * (hostToTarget.so3() * hostVec);
  jacobian.dhost.dp_dq = jacobian.dpi * dHostToTarget.daction_dq_host(hostVecH);
  jacobian.dhost.dp_dt = jacobian.dpi * dHostToTarget.daction_dt_host;
  jacobian.dtarget.dp_dq =
      jacobian.dpi * dHostToTarget.daction_dq_target(hostVecH);
  jacobian.dtarget.dp_dt = jacobian.dpi * dHostToTarget.daction_dt_target;

  for (int i = 0; i < settings.residualPattern.pattern().size(); ++i) {
    double d_da =
        lightHostToTarget.ea() * (hostIntensities[i] - lightWorldToHost.b());
    jacobian.dhost.dr_dab[i][0] = d_da;
    jacobian.dhost.dr_dab[i][1] = lightHostToTarget.ea();
    jacobian.dtarget.dr_dab[i][0] = -d_da;
    jacobian.dtarget.dr_dab[i][1] = -1;
  }

  return jacobian;
}

inline Mat22t sum_gradab(const static_vector<T, Residual::MPS> &weights,
                         const Vec2t gradItarget[], const Vec2t dr_dab[]) {
  Mat22t sum = Mat22t::Zero();
  for (int i = 0; i < weights.size(); ++i)
    sum += weights[i] * gradItarget[i] * dr_dab[i].transpose();
  return sum;
}

template <bool isSameFrame>
inline Residual::FrameFrameHessian
H_frameframe(const Mat27t &df1_dp_dqt, const Mat27t &df2_dp_dqt,
             const Vec2t df1_dr_dab[], const Vec2t df2_dr_dab[],
             const static_vector<T, Residual::MPS> &weights,
             const Mat22t &sum_wgradgradT, const Mat22t &sum_gradab1,
             const Mat22t &sum_gradab2) {
  Residual::FrameFrameHessian H;
  H.qtqt = df1_dp_dqt.transpose() * sum_wgradgradT * df2_dp_dqt;

  H.qtab = df1_dp_dqt.transpose() * sum_gradab2;

  if constexpr (isSameFrame)
    H.abqt = H.qtab.transpose();
  else
    H.abqt = sum_gradab1.transpose() * df2_dp_dqt;

  Mat22t sum_abab = Mat22t::Zero();
  for (int i = 0; i < weights.size(); ++i)
    sum_abab += weights[i] * df1_dr_dab[i] * df2_dr_dab[i].transpose();
  H.abab = sum_abab;

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
Residual::getDeltaHessian(const static_vector<T, MPS> &values,
                          const static_vector<T, MPS> &weights,
                          const Residual::Jacobian &jacobian) {

  DeltaHessian deltaHessian;

  Mat27t dhost_dp_dqt;
  dhost_dp_dqt << jacobian.dhost.dp_dq, jacobian.dhost.dp_dt;
  Mat27t dtarget_dp_dqt;
  dtarget_dp_dqt << jacobian.dtarget.dp_dq, jacobian.dtarget.dp_dt;

  Mat22t sum_wgradgradT;
  for (int i = 0; i < values.size(); ++i)
    sum_wgradgradT += weights[i] * jacobian.gradItarget[i] *
                      jacobian.gradItarget[i].transpose();
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

Residual::FrameFrameHessian::FrameFrameHessian()
  : qtqt(Mat77t::Zero())
  , qtab(Mat72t::Zero())
  , abqt(Mat27t::Zero())
  , abab(Mat22t::Zero()) {}

Residual::FrameFrameHessian &Residual::FrameFrameHessian::operator+=(
    const Residual::FrameFrameHessian &other) {
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

Residual::FramePointHessian &Residual::FramePointHessian::operator+=(
    const Residual::FramePointHessian &other) {
  qtd += other.qtd;
  abd += other.abd;
  return *this;
}

Residual::DeltaHessian::DeltaHessian()
  : pointPoint(0) {}

std::ostream &operator<<(std::ostream &os, const Residual &res) {
  os << "host ind = " << res.hostInd()
     << "\nhost cam ind = " << res.hostCamInd()
     << "\ntarget ind = " << res.targetInd()
     << "\ntarget cam ind = " << res.targetCamInd()
     << "\npoint ind = " << res.pointInd() << "\nlog(depth) = " << *res.logDepth
     << "\nhost point = " << res.hostPoint.transpose()
     << "\nhost dir = " << res.hostDir.transpose() << "\n";
  return os;
}

// std::ostream &operator<<(std::ostream &os, const Residual::Jacobian
// &jacobian) {
//  os << "d(pi)/dp = \n"
//     << jacobian.dpi << "\ndp/dq(host) = \n"
//     << jacobian.dhost.dp_dq << "\ndp/dt(host) = \n"
//     << jacobian.dhost.dp_dt << "\ndp/dq(target) = \n"
//     << jacobian.dtarget.dp_dq << "\ndp/dt(target) = \n"
//     << jacobian.dtarget.dp_dt;
//}

MatR4t Residual::Jacobian::dr_dq_host(int patternSize) const {
  MatR4t dr_dq_host(patternSize, 4);
  for (int i = 0; i < patternSize; ++i)
    dr_dq_host.row(i) = gradItarget[i].transpose() * dhost.dp_dq;
  return dr_dq_host;
}

MatR3t Residual::Jacobian::dr_dt_host(int patternSize) const {
  MatR3t dr_dt_host(patternSize, 3);
  for (int i = 0; i < patternSize; ++i)
    dr_dt_host.row(i) = gradItarget[i].transpose() * dhost.dp_dt;
  return dr_dt_host;
}

MatR4t Residual::Jacobian::dr_dq_target(int patternSize) const {
  MatR4t dr_dq_target(patternSize, 4);
  for (int i = 0; i < patternSize; ++i)
    dr_dq_target.row(i) = gradItarget[i].transpose() * dtarget.dp_dq;
  return dr_dq_target;
}

MatR3t Residual::Jacobian::dr_dt_target(int patternSize) const {
  MatR3t dr_dt_target(patternSize, 3);
  for (int i = 0; i < patternSize; ++i)
    dr_dt_target.row(i) = gradItarget[i].transpose() * dtarget.dp_dt;
  return dr_dt_target;
}

MatR2t Residual::Jacobian::dr_daff_host(int patternSize) const {
  MatR2t dr_daff_host(patternSize, 2);
  for (int i = 0; i < patternSize; ++i)
    dr_daff_host.row(i) = dhost.dr_dab[i].transpose();
  return dr_daff_host;
}

MatR2t Residual::Jacobian::dr_daff_target(int patternSize) const {
  MatR2t dr_daff_target(patternSize, 2);
  for (int i = 0; i < patternSize; ++i)
    dr_daff_target.row(i) = dtarget.dr_dab[i].transpose();
  return dr_daff_target;
}

VecRt Residual::Jacobian::dr_dlogd(int patternSize) const {
  VecRt dr_dlogd(patternSize);
  for (int i = 0; i < patternSize; ++i)
    dr_dlogd[i] = gradItarget[i].transpose() * dp_dlogd;
  return dr_dlogd;
}

} // namespace mdso::optimize
