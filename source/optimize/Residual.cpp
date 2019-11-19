#include "optimize/Residual.h"
#include "PreKeyFrameEntryInternals.h"

namespace mdso::optimize {

Residual::Residual(CameraBundle::CameraEntry *camHost,
                   CameraBundle::CameraEntry *camTarget, KeyFrameEntry *host,
                   KeyFrameEntry *target, OptimizedPoint *optimizedPoint,
                   const SE3 &hostToTarget, ceres::LossFunction *lossFunction,
                   const ResidualSettings &settings)
    : camHost(camHost)
    , camTarget(camTarget)
    , host(host)
    , target(target)
    , optimizedPoint(optimizedPoint)
    , settings(settings)
    , reprojPattern(settings.residualPattern.pattern().size())
    , hostIntencities(settings.residualPattern.pattern().size()) {
  double depth = optimizedPoint->depth();
  Vec2 reproj = camTarget->cam.map(
      hostToTarget *
      (depth * camHost->cam.unmap(optimizedPoint->p).normalized()));
  for (int i = 0; i < reprojPattern.size(); ++i) {
    Vec2 r = camTarget->cam.map(
        hostToTarget *
        (depth *
         camHost->cam
             .unmap(optimizedPoint->p + settings.residualPattern.pattern()[i])
             .normalized()));
    reprojPattern[i] = r - reproj;
  }

  PreKeyFrameEntryInternals::Interpolator_t *hostInterp =
      &host->preKeyFrameEntry->internals->interpolator(0);
  for (int i = 0; i < hostIntencities.size(); ++i) {
    Vec2 p = optimizedPoint->p + settings.residualPattern.pattern()[i];
    Vec2 gradIhost;
    hostInterp->Evaluate(p[1], p[0], &hostIntencities[i], &gradIhost[1],
                         &gradIhost[0]);
    double normSq = gradIhost.squaredNorm();
    double c = settings.gradWeighting.c;
    gradWeights[i] = c / std::sqrt(c * c + normSq);
  }
}

static_vector<T, Residual::MPS>
Residual::getValues(const SE3t &hostToTarget,
                    const AffLightT &lightHostToTarget) {
  static_vector<T, Residual::MPS> result(
      settings.residualPattern.pattern().size());
  T depth = optimizedPoint->depth();
  PreKeyFrameEntryInternals::Interpolator_t *targetInterp =
      &target->preKeyFrameEntry->internals->interpolator(0);
  Vec2t reproj =
      camTarget->cam.map(hostToTarget * (depth * optimizedPoint->dir));
  for (int i = 0; i < result.size(); ++i) {
    Vec2t p = reproj + reprojPattern[i];
    double targetIntencity = INF;
    targetInterp->Evaluate(p[1], p[0], &targetIntencity);
    T hostIntencity = lightHostToTarget(hostIntencities[i]);
    result[i] = T(targetIntencity) - hostIntencity;
  }

  return result;
}

static_vector<T, Residual::MPS>
Residual::getWeights(const static_vector<T, MPS> &values) {
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

Residual::Jacobian Residual::getJacobian(const SE3 &hostToTarget,
                                         const AffLightT &lightHostToTarget,
                                         const Mat33t &worldToTargetRot) {
  const SE3 &hostToWorld = host->host->thisToWorld();
  const SE3 &targetToWorld = target->host->thisToWorld();
  Jacobian jacobian;
  PreKeyFrameEntryInternals::Interpolator_t *targetInterp =
      &target->preKeyFrameEntry->internals->interpolator(0);

  T depth = optimizedPoint->depth();
  Vec3t hostVec = depth * optimizedPoint->dir;
  Vec3t targetVec = hostToTarget * hostVec;
  auto [reproj, piJacobian] = camTarget->cam.diffMap(targetVec);

  for (int i = 0; i < settings.residualPattern.pattern().size(); ++i) {
    Vec2t p = reproj + reprojPattern[i];
    double intensity = INF;
    targetInterp->Evaluate(p[1], p[0], &intensity, &jacobian.gradItarget[i][1],
                           &jacobian.gradItarget[i][0]);
  }

  jacobian.dp_dlogd = piJacobian * (hostToTarget.so3() * hostVec);

  Mat23 tmp1 = piJacobian * worldToTargetRot;
  jacobian.dhost.dp_dqt.leftCols<4>() =
      tmp1 * dRv_dq(hostToWorld.so3(), hostVec);
  jacobian.dhost.dp_dqt.rightCols<3>() = tmp1;

  jacobian.dtarget.dp_dqt = piJacobian * dinvv_dqt(targetToWorld, targetVec);

  for (int i = 0; i < settings.residualPattern.pattern().size(); ++i) {
    double d_da = lightHostToTarget.data[0] * hostIntencities[i];
    jacobian.dhost.dr_dab[i][0] = d_da;
    jacobian.dhost.dr_dab[i][1] = lightHostToTarget.data[0];
    jacobian.dtarget.dr_dab[i][0] = -d_da;
    jacobian.dtarget.dr_dab[i][1] = -1;
  }

  return jacobian;
}

Mat22t sum_gradab(const static_vector<T, Residual::MPS> &weights,
                  const Vec2 gradItarget[], const Vec2 dr_dab[]) {
  Mat22t sum = Mat22t::Zero();
  for (int i = 0; i < weights.size(); ++i)
    sum += weights[i] * gradItarget[i] * dr_dab[i].transpose();
  return sum;
}

Residual::DeltaHessian::FrameFrame
H_frameframe(const Residual::Jacobian::DiffFrameParams &df1,
             const Residual::Jacobian::DiffFrameParams &df2,
             const static_vector<T, Residual::MPS> &weights,
             const Mat22t &sum_wgradgradT, const Mat22t &sum_gradab) {
  Residual::DeltaHessian::FrameFrame H;
  H.qtqt = df1.dp_dqt.transpose() * sum_wgradgradT * df2.dp_dqt;

  H.qtab = df1.dp_dqt.transpose() * sum_gradab;

  Mat22t sum_abab = Mat22t::Zero();
  for (int i = 0; i < weights.size(); ++i)
    sum_abab += weights[i] * df1.dr_dab[i] * df2.dr_dab[i].transpose();
  H.abab = sum_abab;

  return H;
}

Residual::DeltaHessian::FramePoint H_framepoint(const Mat27t &dp_dqt,
                                                const Vec2t &dp_dlogd,
                                                const Mat22t &sum_wgradgradT,
                                                const Mat22t &sum_gradab) {
  Residual::DeltaHessian::FramePoint H;
  H.abd = sum_gradab.transpose() * dp_dlogd;
  H.qtd = dp_dqt.transpose() * sum_wgradgradT * dp_dlogd;
  return H;
}

Residual::DeltaHessian
Residual::getDeltaHessian(const Residual::Jacobian &jacobian,
                          const SE3t &hostToTarget,
                          const AffLightT &lightWorldToTarget) {
  static_vector<T, MPS> values = getValues(hostToTarget, lightWorldToTarget);
  static_vector<T, MPS> weights = getWeights(values);

  DeltaHessian deltaHessian;
  Mat22t sum_wgradgradT;
  for (int i = 0; i < values.size(); ++i)
    sum_wgradgradT += weights[i] * jacobian.gradItarget[i] *
                      jacobian.gradItarget[i].transpose();
  Mat22t sum_gradab_host =
      sum_gradab(weights, jacobian.gradItarget, jacobian.dhost.dr_dab);
  Mat22t sum_gradab_target =
      sum_gradab(weights, jacobian.gradItarget, jacobian.dtarget.dr_dab);
  deltaHessian.hostHost = H_frameframe(jacobian.dhost, jacobian.dhost, weights,
                                       sum_wgradgradT, sum_gradab_host);
  deltaHessian.hostTarget =
      H_frameframe(jacobian.dhost, jacobian.dtarget, weights, sum_wgradgradT,
                   sum_gradab_target);
  deltaHessian.targetTarget =
      H_frameframe(jacobian.dtarget, jacobian.dtarget, weights, sum_wgradgradT,
                   sum_gradab_target);

  deltaHessian.hostPoint =
      H_framepoint(jacobian.dhost.dp_dqt, jacobian.dp_dlogd, sum_wgradgradT,
                   sum_gradab_host);
  deltaHessian.targetPoint =
      H_framepoint(jacobian.dtarget.dp_dqt, jacobian.dp_dlogd, sum_wgradgradT,
                   sum_gradab_target);

  deltaHessian.pointPoint =
      jacobian.dp_dlogd.dot(sum_wgradgradT * jacobian.dp_dlogd);

  return deltaHessian;
}

} // namespace mdso::optimize
