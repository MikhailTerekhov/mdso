#include "optimize/Residual.h"
#include "PreKeyFrameEntryInternals.h"

namespace mdso::optimize {

Residual::Residual(CameraBundle::CameraEntry *camHost,
                   CameraBundle::CameraEntry *camTarget, KeyFrameEntry *host,
                   KeyFrameEntry *targetFrame, OptimizedPoint *optimizedPoint,
                   const SE3 &hostToTargetImage,
                   ceres::LossFunction *lossFunction,
                   const ResidualSettings &settings)
    : lossFunction(lossFunction)
    , camHost(camHost)
    , camTarget(camTarget)
    , host(host)
    , target(targetFrame)
    , optimizedPoint(optimizedPoint)
    , targetFrame(targetFrame)
    , settings(settings)
    , reprojPattern(settings.residualPattern.pattern().size())
    , hostIntensities(settings.residualPattern.pattern().size())
    , gradWeights(settings.residualPattern.pattern().size()) {
  double depth = optimizedPoint->depth();
  Vec2 reproj = camTarget->cam.map(
      hostToTargetImage *
      (depth * camHost->cam.unmap(optimizedPoint->p).normalized()));
  for (int i = 0; i < reprojPattern.size(); ++i) {
    Vec2 r = camTarget->cam.map(
        hostToTargetImage *
        (depth *
         camHost->cam
             .unmap(optimizedPoint->p + settings.residualPattern.pattern()[i])
             .normalized()));
    reprojPattern[i] = r - reproj;
  }

  PreKeyFrameEntryInternals::Interpolator_t *hostInterp =
      &host->preKeyFrameEntry->internals->interpolator(0);
  for (int i = 0; i < hostIntensities.size(); ++i) {
    Vec2 p = optimizedPoint->p + settings.residualPattern.pattern()[i];
    Vec2 gradIhost;
    hostInterp->Evaluate(p[1], p[0], &hostIntensities[i], &gradIhost[1],
                         &gradIhost[0]);
    double normSq = gradIhost.squaredNorm();
    double c = settings.gradWeighting.c;
    gradWeights[i] = c / std::sqrt(c * c + normSq);
  }
}

static_vector<T, Residual::MPS>
Residual::getValues(const SE3 &hostToTargetImage,
                    const AffLightT &lightHostToTarget, Vec2 *reprojOut) const {
  auto &targetInterp =
      targetFrame->preKeyFrameEntry->internals->interpolator(0);
  static_vector<T, Residual::MPS> result(
      settings.residualPattern.pattern().size());
  T depth = optimizedPoint->depth();
  Vec2 reproj =
      camTarget->cam.map(hostToTargetImage * (depth * optimizedPoint->dir));
  for (int i = 0; i < result.size(); ++i) {
    Vec2 p = reproj + reprojPattern[i];
    double targetIntensity = INF;
    targetInterp.Evaluate(p[1], p[0], &targetIntensity);
    T hostIntensity = lightHostToTarget(hostIntensities[i]);
    result[i] = T(targetIntensity) - hostIntensity;
  }

  if (reprojOut)
    *reprojOut = reproj;

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
    weights[i] = gradWeights[i] * (rho[1] + 2 * rho[2] * v2);
  }
  return weights;
}

// differentiate rotation action w.r.t. quaternion
Mat34 dRv_dq(const SO3 &R, const Vec3 &v) {
  const SO3::QuaternionType &q = R.unit_quaternion();
  Mat34 d_dq;
  d_dq.col(3) = 2 * (q.w() * v + q.vec().cross(v));
  Mat33 vq = v * q.vec().transpose();
  Mat33 M = vq.transpose();
  M.diagonal() *= 2;
  d_dq.leftCols<3>() = 2 * (M - q.w() * SO3::hat(q.vec()) - 2 * vq);
  return d_dq;
}

// differentiate inverse SE3 action w.r.t. quaternion and translation
Mat37 dinvv_dqt(const SE3 &Rt, const Vec3 &v) {
  Mat37 d_dqt;
  d_dqt.leftCols<4>() = dRv_dq(Rt.so3(), v - Rt.translation());
  d_dqt.leftCols<3>() *= -1;
  d_dqt.rightCols<3>() = -Rt.rotationMatrix().transpose();
  return d_dqt;
}

Residual::Jacobian
Residual::getJacobian(const SE3t &hostToTarget,
                      const MotionDerivatives &dHostToTarget,
                      const AffLightT &lightWorldToHost,
                      const AffLightT &lightHostToTarget) const {
  Jacobian jacobian;
  PreKeyFrameEntryInternals::Interpolator_t *targetInterp =
      &target->preKeyFrameEntry->internals->interpolator(0);

  T depth = optimizedPoint->depth();
  Vec3t hostVec = (depth * optimizedPoint->dir).cast<T>();
  Vec4t hostVecH = makeHomogeneous(hostVec);
  Vec3t targetVec = hostToTarget * hostVec;
  auto [reproj, piJacobian] = camTarget->cam.diffMap(targetVec.cast<double>());
  jacobian.dpi = piJacobian.cast<T>();

  for (int i = 0; i < settings.residualPattern.pattern().size(); ++i) {
    Vec2 p = reproj + reprojPattern[i];
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

inline Residual::DeltaHessian::FrameFrame
H_frameframe(const Mat27t &df1_dp_dqt, const Mat27t &df2_dp_dqt,
             const Vec2t df1_dr_dab[], const Vec2t df2_dr_dab[],
             const static_vector<T, Residual::MPS> &weights,
             const Mat22t &sum_wgradgradT, const Mat22t &sum_gradab) {
  Residual::DeltaHessian::FrameFrame H;
  H.qtqt = df1_dp_dqt.transpose() * sum_wgradgradT * df2_dp_dqt;

  H.qtab = df1_dp_dqt.transpose() * sum_gradab;

  Mat22t sum_abab = Mat22t::Zero();
  for (int i = 0; i < weights.size(); ++i)
    sum_abab += weights[i] * df1_dr_dab[i] * df2_dr_dab[i].transpose();
  H.abab = sum_abab;

  return H;
}

inline Residual::DeltaHessian::FramePoint
H_framepoint(const Mat27t &dp_dqt, const Vec2t &dp_dlogd,
             const Mat22t &sum_wgradgradT, const Mat22t &sum_gradab) {
  Residual::DeltaHessian::FramePoint H;
  H.abd = sum_gradab.transpose() * dp_dlogd;
  H.qtd = dp_dqt.transpose() * sum_wgradgradT * dp_dlogd;
  return H;
}

Residual::DeltaHessian Residual::getDeltaHessian(
    const Residual::Jacobian &jacobian, const MotionDerivatives &dHostToTarget,
    const SE3 &hostToTarget, const AffLightT &lightWorldToTarget) const {
  static_vector<T, MPS> values = getValues(hostToTarget, lightWorldToTarget);
  static_vector<T, MPS> weights = getWeights(values);

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

  deltaHessian.hostHost = H_frameframe(
      dhost_dp_dqt, dhost_dp_dqt, jacobian.dhost.dr_dab, jacobian.dhost.dr_dab,
      weights, sum_wgradgradT, sum_gradab_host);
  deltaHessian.hostTarget = H_frameframe(
      dhost_dp_dqt, dtarget_dp_dqt, jacobian.dhost.dr_dab,
      jacobian.dtarget.dr_dab, weights, sum_wgradgradT, sum_gradab_target);
  deltaHessian.targetTarget = H_frameframe(
      dtarget_dp_dqt, dtarget_dp_dqt, jacobian.dtarget.dr_dab,
      jacobian.dtarget.dr_dab, weights, sum_wgradgradT, sum_gradab_target);

  deltaHessian.hostPoint = H_framepoint(dhost_dp_dqt, jacobian.dp_dlogd,
                                        sum_wgradgradT, sum_gradab_host);
  deltaHessian.targetPoint = H_framepoint(dtarget_dp_dqt, jacobian.dp_dlogd,
                                          sum_wgradgradT, sum_gradab_target);

  deltaHessian.pointPoint =
      jacobian.dp_dlogd.dot(sum_wgradgradT * jacobian.dp_dlogd);

  return deltaHessian;
}

} // namespace mdso::optimize
