#include "optimize/MotionDerivatives.h"
#include "util/util.h"
#include <ceres/jet.h>
#include <gtest/gtest.h>

using namespace mdso;
using namespace mdso::optimize;

using JetSE3 = ceres::Jet<T, SE3t::num_parameters>;
using Vec3J = Eigen::Matrix<JetSE3, 3, 1, Eigen::DontAlign>;
using Mat34J = Eigen::Matrix<JetSE3, 3, 4, Eigen::DontAlign>;
using SE3J = Sophus::SE3<JetSE3, Eigen::DontAlign>;
using SO3J = Sophus::SO3<JetSE3, Eigen::DontAlign>;
using ArrayDQ = std::array<Mat34t, SO3::num_parameters>;

std::mt19937 mt;

Mat12x4t toJacobian(const ArrayDQ &dq) {
  Mat12x4t jacobian;
  for (int qi = 0; qi < SO3t::num_parameters; ++qi)
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 4; ++c)
        jacobian(r * 4 + c, qi) = dq[qi](r, c);
  return jacobian;
}

template <int nret>
T tangentErr(const Eigen::Matrix<T, nret, 4> &expected,
             const Eigen::Matrix<T, nret, 4> &actual, const SO3t &diffRotation,
             Eigen::Matrix<T, nret, 3> &expectedTang,
             Eigen::Matrix<T, nret, 3> &actualTang) {
  Mat43t dParam = diffRotation.Dx_this_mul_exp_x_at_0();
  expectedTang = expected * dParam;
  actualTang = actual * dParam;
  return (expectedTang - actualTang).norm();
}

SE3J compose(const SE3J &hostFrameToBody, const SE3J &hostBodyToWorld,
             const SE3J &targetBodyToWorld, const SE3J &targetBodyToFrame) {
  return targetBodyToFrame * targetBodyToWorld.inverse() * hostBodyToWorld *
         hostFrameToBody;
}

SE3J idJet(const SE3J &motion) {
  SE3J result = motion;
  for (int i = 0; i < SO3::num_parameters; ++i)
    result.so3().data()[i].v[i] = 1;
  for (int i = 0; i < 3; ++i)
    result.translation().data()[i].v[4 + i] = 1;
  return result;
}

ArrayDQ d_dq(const SE3J &composeRes) {
  Mat34J mat = composeRes.matrix3x4();
  ArrayDQ dmatrix_dq;
  for (int qi = 0; qi < SO3t::num_parameters; ++qi)
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 4; ++c)
        dmatrix_dq[qi](r, c) = mat(r, c).v[qi];
  return dmatrix_dq;
}

Mat33t d_dt(const SE3J &composeRes, const Vec3t &v) {
  Vec3J vJ = v.cast<JetSE3>();
  Vec3J movedV = composeRes * vJ;
  Mat33t res;
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c)
      res(r, c) = movedV(r).v[4 + c];
  return res;
}

struct MotionDerivativesInp {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  SE3t hostFrameToBody;
  SE3t hostBodyToWorld;
  SE3t targetBodyToWorld;
  SE3t targetBodyToFrame;

  friend std::ostream &operator<<(std::ostream &out,
                                  const MotionDerivativesInp &data) {
    putInMatrixForm(out, data.hostFrameToBody.cast<double>());
    putInMatrixForm(out, data.hostBodyToWorld.cast<double>());
    putInMatrixForm(out, data.targetBodyToWorld.cast<double>());
    putInMatrixForm(out, data.targetBodyToFrame.cast<double>());

    return out;
  }
};

template <typename T> struct ErrorBounds;

template <> struct ErrorBounds<float> {
  static constexpr float dq = 5e-6;
  static constexpr float dt = 5e-6;
  static constexpr float dAction = 5e-6;
};

template <> struct ErrorBounds<double> {
  static constexpr float dq = 1e-10;
  static constexpr float dt = 1e-10;
  static constexpr float dAction = 1e-10;
};

class MotionDerivativesTest
    : public ::testing::TestWithParam<MotionDerivativesInp> {
protected:
  void SetUp() {
    hostFrameToBody = GetParam().hostFrameToBody.cast<JetSE3>();
    targetBodyToFrame = GetParam().targetBodyToFrame.cast<JetSE3>();
    hostBodyToWorld = GetParam().hostBodyToWorld.cast<JetSE3>();
    targetBodyToWorld = GetParam().targetBodyToWorld.cast<JetSE3>();
    hostBodyToWorldPert = idJet(hostBodyToWorld);
    targetBodyToWorldPert = idJet(targetBodyToWorld);

    composeHost = compose(hostFrameToBody, hostBodyToWorldPert,
                          targetBodyToWorld, targetBodyToFrame);
    composeTarget = compose(hostFrameToBody, hostBodyToWorld,
                            targetBodyToWorldPert, targetBodyToFrame);

    derivatives = std::unique_ptr<MotionDerivatives>(new MotionDerivatives(
        GetParam().hostFrameToBody, GetParam().hostBodyToWorld,
        GetParam().targetBodyToWorld, GetParam().targetBodyToFrame));
  }

  static constexpr T dqErr = ErrorBounds<T>::dq;
  static constexpr T dtErr = ErrorBounds<T>::dt;
  static constexpr T dActionErr = ErrorBounds<T>::dAction;

  SE3J hostFrameToBody;
  SE3J targetBodyToFrame;
  SE3J hostBodyToWorld;
  SE3J hostBodyToWorldPert;
  SE3J targetBodyToWorld;
  SE3J targetBodyToWorldPert;

  SE3J composeHost;
  SE3J composeTarget;

  std::unique_ptr<MotionDerivatives> derivatives;
};

TEST_P(MotionDerivativesTest, CorrectDDtHost) {
  std::uniform_real_distribution<double> d(-1, 1);
  Vec3t v = Vec3t(d(mt), d(mt), d(mt));
  Mat33t d_dt_host = d_dt(composeHost, v);
  double err = (d_dt_host - derivatives->d_dt_host).norm();
  ASSERT_LE(err, dtErr) << "d_dt_host big error (=" << err << ")\n"
                        << "actual =\n"
                        << derivatives->d_dt_host << "\nexpected =\n"
                        << d_dt_host << "\nparams:\n"
                        << GetParam();
}

TEST_P(MotionDerivativesTest, CorrectDDtTarget) {
  std::uniform_real_distribution<double> d(-1, 1);
  Vec3t v = Vec3t(d(mt), d(mt), d(mt));
  Mat33t d_dt_target = d_dt(composeTarget, v);
  T err = (d_dt_target - derivatives->d_dt_target).norm();
  ASSERT_LE(err, dtErr) << "d_dt_host big error (=" << err << ")\n"
                        << "actual =\n"
                        << derivatives->d_dt_host << "\nexpected =\n"
                        << d_dt_target << "\nparams:\n"
                        << GetParam();
}

TEST_P(MotionDerivativesTest, CorrectDDqHost) {
  auto dmatrix_dq_host = d_dq(composeHost);

  auto testInfo = ::testing::UnitTest::GetInstance()->current_test_info();
  std::string caseName(testInfo->test_case_name());
  std::string testName(testInfo->name());

  Mat12x3t d_dq_tangent_actual, d_dq_tangent_expected;
  ArrayDQ actual;
  std::copy(derivatives->dmatrix_dqi_host, derivatives->dmatrix_dqi_host + 4,
            actual.begin());
  T err = tangentErr(toJacobian(dmatrix_dq_host), toJacobian(actual),
                     GetParam().hostBodyToWorld.so3(), d_dq_tangent_expected,
                     d_dq_tangent_actual);
  ASSERT_LE(err, dqErr) << "d_dq_host big error (=" << err << ")\n"
                        << "actual =\n"
                        << d_dq_tangent_actual << "\nexpected =\n"
                        << d_dq_tangent_expected << "\nparams:\n"
                        << GetParam();
}

TEST_P(MotionDerivativesTest, CorrectDDqTarget) {
  auto dmatrix_dq_target = d_dq(composeTarget);
  Mat12x3t d_dq_tangent_actual, d_dq_tangent_expected;
  ArrayDQ actual;
  std::copy(derivatives->dmatrix_dqi_target,
            derivatives->dmatrix_dqi_target + 4, actual.begin());
  T err = tangentErr(toJacobian(dmatrix_dq_target), toJacobian(actual),
                     GetParam().targetBodyToWorld.so3(), d_dq_tangent_expected,
                     d_dq_tangent_actual);
  ASSERT_LE(err, dqErr) << "d_dq_target big error (=" << err << ")\n"
                        << "actual =\n"
                        << d_dq_tangent_actual << "\nexpected =\n"
                        << d_dq_tangent_expected << "\nparams:\n"
                        << GetParam();
}

TEST_P(MotionDerivativesTest, CorrectDiffActionHost) {
  constexpr int testCount = 10;
  std::uniform_real_distribution<double> d(-1, 1);
  for (int it = 0; it < testCount; ++it) {
    Vec3t v(d(mt), d(mt), d(mt));
    Vec4t vH = makeHomogeneous(v);
    Vec3J moved = composeHost * v.cast<JetSE3>();
    Mat34t expected;
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 4; ++c)
        expected(r, c) = moved[r].v[c];
    Mat34t actual = derivatives->diffActionHostQ(vH);
    Mat33t expectedTang, actualTang;
    T err = tangentErr(expected, actual, GetParam().hostBodyToWorld.so3(),
                       expectedTang, actualTang);
    ASSERT_LE(err, dActionErr) << "d_dq_host big error (=" << err << ")\n"
                          << "actual =\n"
                          << actualTang << "\nexpected =\n"
                          << expectedTang << "\nparams:\n"
                          << GetParam();
  }
}

TEST_P(MotionDerivativesTest, CorrectDiffActionTarget) {
  constexpr int testCount = 10;
  std::uniform_real_distribution<double> d(-1, 1);
  for (int it = 0; it < testCount; ++it) {
    Vec3t v(d(mt), d(mt), d(mt));
    Vec4t vH = makeHomogeneous(v);
    Vec3J moved = composeTarget * v.cast<JetSE3>();
    Mat34t expected;
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 4; ++c)
        expected(r, c) = moved[r].v[c];
    Mat34t actual = derivatives->diffActionTargetQ(vH);
    Mat33t expectedTang, actualTang;
    T err = tangentErr(expected, actual, GetParam().targetBodyToWorld.so3(),
                       expectedTang, actualTang);
    ASSERT_LE(err, dActionErr) << "d_dq_host big error (=" << err << ")\n"
                               << "actual =\n"
                               << actualTang << "\nexpected =\n"
                               << expectedTang << "\nparams:\n"
                               << GetParam();
  }
}

StdVector<MotionDerivativesInp> hostNId() {
  StdVector<MotionDerivativesInp> tests;
  for (int it = 0; it < 100; ++it)
    tests.push_back({SE3t(), SE3t::sampleUniform(mt), SE3t(), SE3t()});
  return tests;
}

StdVector<MotionDerivativesInp> targetNId() {
  StdVector<MotionDerivativesInp> tests;
  for (int it = 0; it < 100; ++it)
    tests.push_back({SE3t(), SE3t(), SE3t::sampleUniform(mt), SE3t()});
  return tests;
}

StdVector<MotionDerivativesInp> bodyToWorldId() {
  StdVector<MotionDerivativesInp> tests;
  for (int it = 0; it < 100; ++it)
    tests.push_back(
        {SE3t(), SE3t::sampleUniform(mt), SE3t::sampleUniform(mt), SE3t()});
  return tests;
}

StdVector<MotionDerivativesInp> generalCase() {
  StdVector<MotionDerivativesInp> tests;
  for (int it = 0; it < 100; ++it)
    tests.push_back({SE3t::sampleUniform(mt), SE3t::sampleUniform(mt),
                     SE3t::sampleUniform(mt), SE3t::sampleUniform(mt)});
  return tests;
}

INSTANTIATE_TEST_CASE_P(
    AllId, MotionDerivativesTest,
    testing::Values<MotionDerivativesInp>({SE3t(), SE3t(), SE3t(), SE3t()}));

INSTANTIATE_TEST_CASE_P(HostNotId, MotionDerivativesTest,
                        testing::ValuesIn(hostNId()));

INSTANTIATE_TEST_CASE_P(TargetNotId, MotionDerivativesTest,
                        testing::ValuesIn(targetNId()));

INSTANTIATE_TEST_CASE_P(BodyToWorldId, MotionDerivativesTest,
                        testing::ValuesIn(bodyToWorldId()));

INSTANTIATE_TEST_CASE_P(GeneralCase, MotionDerivativesTest,
                        testing::ValuesIn(generalCase()));

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
