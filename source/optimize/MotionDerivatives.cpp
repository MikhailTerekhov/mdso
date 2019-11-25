#include "optimize/MotionDerivatives.h"
#include <glog/logging.h>

namespace mdso::optimize {

std::array<Mat33t, SO3t::num_parameters> diffQToMatrix(const SO3t &rot) {
  auto &q = rot.unit_quaternion();
  auto &u = q.vec();
  // clang-format off
  return {2 * (Mat33t() << 0   , u[1]     , u[2]     ,
                           u[1], -2 * u[0], -q.w()   ,
                           u[2], q.w()    , -2 * u[0]
              ).finished(),
          2 * (Mat33t() << -2 * u[1], u[0], q.w()    ,
                           u[0]     , 0   , u[2]     ,
                           -q.w()   , u[2], -2 * u[1]
              ).finished(),
          2 * (Mat33t() << -2 * u[2], -q.w()   , u[0],
                           q.w()    , -2 * u[2], u[1],
                           u[0]     , u[1]     , 0
              ).finished(),
          2 * SO3t::hat(u)
  };
  // clang-format on
}

MotionDerivatives::MotionDerivatives(const SE3t &hostFrameToBody,
                                     const SE3t &hostBodyToWorld,
                                     const SE3t &targetBodyToWorld,
                                     const SE3t &targetBodyToFrame) {
  SE3t targetWorldToBody = targetBodyToWorld.inverse();

  d_dt_host = (targetBodyToFrame.so3() * targetWorldToBody.so3()).matrix();
  d_dt_target = -d_dt_host;

  Mat44t hostLeft = (targetBodyToFrame * targetWorldToBody).matrix();
  Mat44t hostRight = hostFrameToBody.matrix();

  auto dmatrixrot_dqi_host = diffQToMatrix(hostBodyToWorld.so3());
  for (int i = 0; i < SO3t::num_parameters; ++i) {
    Mat44t dmatrix4x4_dqi_host = Mat44t::Zero();
    dmatrix4x4_dqi_host.topLeftCorner<3, 3>() = dmatrixrot_dqi_host[i];
    dmatrix4x4_dqi_host = hostLeft * dmatrix4x4_dqi_host * hostRight;
    CHECK_LE(dmatrix4x4_dqi_host.row(3).norm(), 1e-7);
    dmatrix_dqi_host[i] = dmatrix4x4_dqi_host.topRows<3>();
  }

  Mat44t targetLeft = targetBodyToFrame.matrix();
  Mat44t targetRight = (hostBodyToWorld * hostFrameToBody).matrix();

  auto dmatrixrot_dqi_target = diffQToMatrix(targetWorldToBody.so3().inverse());
  for (int i = 0; i < SO3t::num_parameters; ++i) {
    dmatrixrot_dqi_target[i].transposeInPlace();
    Mat44t dmatrix4x4_dqi_target = Mat44t::Zero();
    dmatrix4x4_dqi_target.topLeftCorner<3, 3>() = dmatrixrot_dqi_target[i];
    dmatrix4x4_dqi_target.topRightCorner<3, 1>() =
        -(dmatrixrot_dqi_target[i] * targetBodyToWorld.translation());
    dmatrix4x4_dqi_target = targetLeft * dmatrix4x4_dqi_target * targetRight;
    CHECK_LE(dmatrix4x4_dqi_target.row(3).norm(), 1e-7);
    dmatrix_dqi_target[i] = dmatrix4x4_dqi_target.topRows<3>();
  }
}

Mat34t MotionDerivatives::diffActionQ(const Mat34t dmatrix_dq[], const Vec4t &vH) {
  Mat34t result;
  for (int i = 0; i < 4; ++i)
    result.col(i) = dmatrix_dq[i] * vH;
  return result;
}

} // namespace mdso::optimize
