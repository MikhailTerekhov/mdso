#include "optimize/parametrizations.h"
#include <ceres/numeric_diff_cost_function.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

using namespace mdso;
using namespace mdso::optimize;

constexpr int sampleCount = 100;

template <typename T> struct ErrorBounds;

template <> struct ErrorBounds<float> {
  static constexpr float zeroAddRelDelta = 1e-7;
  static constexpr float jacobianRelDelta = 1e-7;

  template <typename Parametrization> struct OnManifold;
};

template <>
struct ErrorBounds<float>::OnManifold<RightExpParametrization<SO3t>> {
  static constexpr float value = 1e-8;
};
template <>
struct ErrorBounds<float>::OnManifold<RightExpParametrization<SE3t>> {
  static constexpr float value = 1e-8;
};
template <> struct ErrorBounds<float>::OnManifold<S2Parametrization> {
  static constexpr float value = 1e-8;
};
template <> struct ErrorBounds<float>::OnManifold<SO3xS2Parametrization> {
  static constexpr float value = 1e-8;
};

template <> struct ErrorBounds<double> {
  static constexpr double zeroAddRelDelta = 1e-14;
  static constexpr float jacobianRelDelta = 1e-13;

  template <typename Parametrization> struct OnManifold;
};

template <typename UniformBitGenerator>
std::pair<Vec3t, Vec3t> sampleSphere(UniformBitGenerator &g) {
  std::uniform_real_distribution<T> coord;
  Vec3t center, init;
  do {
    center = Vec3t(coord(g), coord(g), coord(g));
    init = Vec3t(coord(g), coord(g), coord(g));
  } while ((init - center).norm() < 1e-2);
  return {center, init};
}

template <>
struct ErrorBounds<double>::OnManifold<RightExpParametrization<SO3t>> {
  static constexpr float value = 1e-14;
};
template <>
struct ErrorBounds<double>::OnManifold<RightExpParametrization<SE3t>> {
  static constexpr float value = 1e-14;
};
template <> struct ErrorBounds<double>::OnManifold<S2Parametrization> {
  static constexpr float value = 1e-14;
};
template <> struct ErrorBounds<double>::OnManifold<SO3xS2Parametrization> {
  static constexpr float value = 1e-14;
};

template <typename Parametrization, typename UniformBitGenerator>
Parametrization sample(UniformBitGenerator &g) {
  return Parametrization(sampleUniform(g));
}

template <>
RightExpParametrization<SO3t>
sample<RightExpParametrization<SO3t>, std::mt19937>(std::mt19937 &g) {
  return RightExpParametrization<SO3t>(SO3t::sampleUniform(g));
}

template <>
RightExpParametrization<SE3t>
sample<RightExpParametrization<SE3t>, std::mt19937>(std::mt19937 &g) {
  return RightExpParametrization<SE3t>(SE3t::sampleUniform(g));
}

template <>
S2Parametrization sample<S2Parametrization, std::mt19937>(std::mt19937 &g) {
    auto [center, init] = sampleSphere(g);
//  Vec3 center(0, 0, 0);
//  Vec3 init(0, 0, 1);
  return S2Parametrization(center, init);
}

template <>
SO3xS2Parametrization
sample<SO3xS2Parametrization, std::mt19937>(std::mt19937 &g) {
  auto [center, init] = sampleSphere(g);
  return SO3xS2Parametrization(SO3t::sampleUniform(g), center, init);
}

template <typename Parametrization> T errOnManifold(const Parametrization &p);
template <>
T errOnManifold<RightExpParametrization<SO3t>>(
    const RightExpParametrization<SO3t> &p) {
  return std::abs(p.value().unit_quaternion().coeffs().norm() - 1);
}

template <>
T errOnManifold<RightExpParametrization<SE3t>>(
    const RightExpParametrization<SE3t> &p) {
  return std::abs(p.value().unit_quaternion().coeffs().norm() - 1);
}

template <> T errOnManifold<S2Parametrization>(const S2Parametrization &p) {
  return std::abs((p.value() - p.center()).norm() - p.radius());
}

template <>
T errOnManifold<SO3xS2Parametrization>(const SO3xS2Parametrization &p) {
  return std::hypot(errOnManifold<RightExpParametrization<SO3t>>(p.so3()),
                    errOnManifold<S2Parametrization>(p.s2()));
}

template <typename TypeParam>
class ParametrizationTest : public ::testing::Test {
public:
  using Parametrization = TypeParam;
  using VecNPt = Eigen::Matrix<T, Parametrization::num_parameters, 1>;

  void SetUp() override {
    std::mt19937 mt;
    samples.reserve(sampleCount);
    for (int i = 0; i < sampleCount; ++i)
      samples.push_back(sample<Parametrization>(mt));
  }

  StdVector<Parametrization> samples;
};

using Parametrizations =
    ::testing::Types<RightExpParametrization<SO3t>,
                     RightExpParametrization<SE3t>, S2Parametrization,
                     SO3xS2Parametrization>;
TYPED_TEST_CASE(ParametrizationTest, Parametrizations);

template <typename Parametrization> class PlusFunctor {
public:
  PlusFunctor(const Parametrization &p)
      : p(p) {}

  bool operator()(const double *deltaP, double *result) const {
    using Tangent = typename Parametrization::Tangent;
    Parametrization newP = p;
    Eigen::Map<const Tangent> deltaM(deltaP);
    Tangent delta(deltaM);
    newP.addDelta(delta);
    auto value = newP.value();
    for (int i = 0; i < Parametrization::num_parameters; ++i)
      result[i] = double(value.data()[i]);
    return true;
  }

private:
  Parametrization p;
};

TYPED_TEST(ParametrizationTest, diffPlus) {
  using Parametrization = typename TestFixture::Parametrization;
  constexpr int nparams = Parametrization::num_parameters;
  constexpr int DoF = Parametrization::DoF;
  using VecNPt = Eigen::Matrix<T, nparams, 1>;
  using VecDoFt = Eigen::Matrix<T, DoF, 1>;
  using MatDiff = Eigen::Matrix<T, nparams, DoF, Eigen::RowMajor>;

  for (const Parametrization &p : this->samples) {
    ceres::NumericDiffCostFunction<PlusFunctor<Parametrization>, ceres::RIDDERS,
                                   nparams, DoF>
        plus(new PlusFunctor(p));
    MatDiff expectedJacobian;
    VecDoFt delta = VecDoFt::Zero();
    VecNPt pPlus0;
    auto value = p.value();
    Eigen::Map<VecNPt> pRaw(value.data());
    double *expectedJacobianData = expectedJacobian.data();
    double *deltaData = delta.data();
    plus.Evaluate(&deltaData, pPlus0.data(), &expectedJacobianData);
    MatDiff actualJacobian = p.diffPlus();

    T errJacobian =
        (expectedJacobian - actualJacobian).norm() / expectedJacobian.norm();
    ASSERT_LE(errJacobian, ErrorBounds<T>::jacobianRelDelta)
        << "expected =\n"
        << expectedJacobian << "\nactual =\n"
        << actualJacobian;

    T errZeroAdd = (pPlus0 - pRaw).norm() / pRaw.norm();
    ASSERT_LE(errZeroAdd, ErrorBounds<T>::zeroAddRelDelta)
        << "expected =\n"
        << pRaw.transpose() << "\nactual =\n"
        << pPlus0.transpose();
  }
}

TYPED_TEST(ParametrizationTest, remainsOnManifold) {
  using Parametrization = typename TestFixture::Parametrization;
  constexpr int DoF = Parametrization::DoF;
  constexpr T minDelta = -1, maxDelta = 1;
  constexpr int numAdditions = 10;
  using VecDoFt = Eigen::Matrix<T, DoF, 1>;

  std::mt19937 mt;
  std::uniform_real_distribution<T> deltaCoord(minDelta, maxDelta);
  for (const Parametrization &p : this->samples) {
    Parametrization newP = p;
    for (int it = 0; it < numAdditions; ++it) {
      VecDoFt delta;
      for (int j = 0; j < DoF; ++j)
        delta[j] = deltaCoord(mt);
      newP.addDelta(delta);
    }
    double err = errOnManifold<Parametrization>(newP);
    ASSERT_LE(err, ErrorBounds<T>::OnManifold<Parametrization>::value);
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  return RUN_ALL_TESTS();
}
