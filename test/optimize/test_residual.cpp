#include "../../internal/include/PreKeyFrameEntryInternals.h"
#include "../../samples/mfov/reader/MultiFovReader.h"
#include "optimize/Residual.h"
#include "system/CameraBundle.h"
#include "system/IdentityPreprocessor.h"
#include "util/util.h"
#include <ceres/ceres.h>
#include <gtest/gtest.h>
#include <system/FrameTracker.h>

using namespace mdso;
using namespace mdso::optimize;

using MatX2RM = Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>;
using MatX3RM = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
using MatX4RM = Eigen::Matrix<double, Eigen::Dynamic, 4, Eigen::RowMajor>;

DEFINE_string(mfov_dir, "/shared/datasets/mfov",
              "Root folder of the MultiFoV dataset.");

template <typename T> struct ErrorBounds;

template <> struct ErrorBounds<float> {
  static constexpr float resEps = 2;
  static constexpr float valueEps = 5e-4;
  static constexpr float diffRotEps = 5;
  static constexpr float diffTransEps = 1;
  static constexpr float diffAffEps = 1;
  static constexpr float diffDepthEps = 1e-7;
};

template <> struct ErrorBounds<double> {
  static constexpr double resEps = 2;
  static constexpr double valueEps = 5e-10;
  static constexpr double diffRotEps = 5e-6;
  static constexpr double diffTransEps = 5e-6;
  static constexpr double diffAffEps = 5e-10;
  static constexpr double diffDepthEps = 1e-9;
};

cv::Mat3b cvtAff(const cv::Mat3b &mat, const AffLight &aff) {
  cv::Mat3b result;
  cv::convertScaleAbs(mat, result, aff.ea(), aff.b());
  return result;
}

template <typename UniformBitGenerator>
SE3 sampleSe3(double rotDelta, double transDelta, UniformBitGenerator &gen) {
  double t2 = transDelta / 2;
  double r2 = rotDelta / 2;
  std::uniform_real_distribution<double> dtrans(-t2, t2);
  std::uniform_real_distribution<double> drot(-r2, r2);
  Vec3 trans(dtrans(gen), dtrans(gen), dtrans(gen));
  Vec3 rot(drot(gen), drot(gen), drot(gen));
  return SE3(SO3::exp(rot), trans);
}

template <typename UniformBitGenerator>
AffLight sampleAffLight(const Settings::AffineLight &affSettings,
                        UniformBitGenerator &gen) {
  std::uniform_real_distribution<double> da(affSettings.minAffineLightA,
                                            affSettings.maxAffineLightA);
  std::uniform_real_distribution<double> db(affSettings.minAffineLightB,
                                            affSettings.maxAffineLightB);
  return AffLight(da(gen), db(gen));
}

class ResidualTest : public ::testing::Test {
protected:
  void SetUp() override {
    MultiFovReader reader(FLAGS_mfov_dir);
    PixelSelector pixelSelector;

    constexpr int fnum1 = 375, fnum2 = 385;
    Timestamp ts1 = fnum1, ts2 = fnum2;

    SE3 f1ToF2GT = reader.getWorldToFrameGT(fnum2) *
                   reader.getWorldToFrameGT(fnum1).inverse();
    SE3 bodyToCam1 = SE3::sampleUniform(mt);
    SE3 bodyToCam2 = SE3::sampleUniform(mt);
    bodyToCam1.translation() *= f1ToF2GT.translation().norm();
    bodyToCam2.translation() *= f1ToF2GT.translation().norm();
    //  SE3 bodyToCam1;
    //  SE3 bodyToCam2;
    SE3 f1ToF2 = bodyToCam2.inverse() * f1ToF2GT * bodyToCam1;
    cam1.reset(new CameraBundle(&bodyToCam1, reader.cam.get(), 1));
    cam2.reset(new CameraBundle(&bodyToCam2, reader.cam.get(), 1));

    IdentityPreprocessor idPrep;

    cv::Mat3b frame1 = reader.getFrame(fnum1), frame2 = reader.getFrame(fnum2);
    AffLight lightWorldToF1 = sampleAffLight(settings.affineLight, mt);
    AffLight lightWorldToF2 = sampleAffLight(settings.affineLight, mt);
    //    AffLight lightWorldToF1;
    //    AffLight lightWorldToF2;
    frame1 = cvtAff(frame1, lightWorldToF1);
    frame2 = cvtAff(frame2, lightWorldToF2);
    {
      // discard one pixel selection such that the adaptive algorithm will
      // choose the right amount of points for kf1 and kf2
      cv::Mat1d gradX, gradY, gradNorm;
      grad(cvtBgrToGray(frame1), gradX, gradY, gradNorm);
      pixelSelector.select(frame1, gradNorm,
                           settings.keyFrame.immaturePointsNum());
    }

    cv::Mat1d depths1 = reader.getDepths(fnum1),
              depths2 = reader.getDepths(fnum2);
    std::unique_ptr<PreKeyFrame> pkf1(new PreKeyFrame(
        nullptr, cam1.get(), &idPrep, &frame1, 1, &ts1, settings.pyramid));
    kf1.reset(new KeyFrame(std::move(pkf1), &pixelSelector, settings.keyFrame,
                           settings.getPointTracerSettings()));
    kf1->frames[0].lightWorldToThis = lightWorldToF1;
    std::unique_ptr<PreKeyFrame> pkf2(new PreKeyFrame(
        kf1.get(), cam2.get(), &idPrep, &frame2, 2, &ts2, settings.pyramid));
    TrackingResult trackingResult(1);
    trackingResult.baseToTracked = f1ToF2;
    trackingResult.lightBaseToTracked[0] =
        lightWorldToF2 * lightWorldToF1.inverse();
    pkf2->setTracked(trackingResult);
    kf2.reset(new KeyFrame(std::move(pkf2), &pixelSelector, settings.keyFrame,
                           settings.getPointTracerSettings()));

    for (ImmaturePoint &ip : kf1->frames[0].immaturePoints) {
      ip.setTrueDepth(depths1(toCvPoint(ip.p)), settings.pointTracer);
      kf1->frames[0].optimizedPoints.emplace_back(ip);
    }
    kf1->frames[0].immaturePoints.clear();
  }

  static constexpr T resEps = ErrorBounds<T>::resEps;
  static constexpr T valueEps = ErrorBounds<T>::valueEps;
  static constexpr T diffRotEps = ErrorBounds<T>::diffRotEps;
  static constexpr T diffTransEps = ErrorBounds<T>::diffTransEps;
  static constexpr T diffAffEps = ErrorBounds<T>::diffAffEps;
  static constexpr T diffDepthEps = ErrorBounds<T>::diffDepthEps;

  std::mt19937 mt;
  std::unique_ptr<CameraBundle> cam1, cam2;
  std::unique_ptr<KeyFrame> kf1, kf2;
  Settings settings;
};

TEST_F(ResidualTest, IsSmallOnGT) {
  static constexpr double expectedPercentileOfSmall = 0.1;

  SE3 hostToTarget = cam2->bundle[0].bodyToThis * kf2->thisToWorld().inverse() *
                     kf1->thisToWorld() * cam1->bundle[0].thisToBody;
  AffineLightTransform<T> lightHostToTarget =
      (kf2->frames[0].lightWorldToThis *
       kf1->frames[0].lightWorldToThis.inverse())
          .cast<T>();

  StdVector<Residual> residuals;
  ceres::TrivialLoss loss;
  for (OptimizedPoint &op : kf1->frames[0].optimizedPoints)
    residuals.emplace_back(&cam1->bundle[0], &cam2->bundle[0], &kf1->frames[0],
                           &kf2->frames[0], &op, hostToTarget, &loss,
                           settings.getResidualSettings());

  std::vector<static_vector<T, Residual::MPS>> resValues;

  int RS = settings.residualPattern.pattern().size();

  for (const Residual &res : residuals) {
    resValues.push_back(res.getValues(hostToTarget, lightHostToTarget));
  }

  int PS = settings.residualPattern.pattern().size();
  static_vector<std::vector<double>, Residual::MPS> values(PS);
  for (const auto &va : resValues)
    for (int i = 0; i < PS; ++i)
      values[i].push_back(va[i]);

  for (int i = 0; i < PS; ++i) {
    int nSmall = std::count_if(values[i].begin(), values[i].end(),
                               [](double v) { return std::abs(v) < resEps; });

    double smallPercentile = double(nSmall) / values[i].size();
    LOG(INFO) << "percentile of small |r| for pattern[" << i
              << "]: " << smallPercentile << '\n';
    ASSERT_GE(smallPercentile, expectedPercentileOfSmall)
        << "Too much non-zero residuals for pattern[" << i << "]";
  }
}

class ResidualCostFunctor {
public:
  ResidualCostFunctor(
      PreKeyFrameEntryInternals::Interpolator_t *targetFrame,
      const SE3 &hostFrameToBody, const SE3 &targetBodyToFrame, const Vec3 &dir,
      const static_vector<Vec2, Residual::MPS> &reprojPattern,
      const static_vector<double, Residual::MPS> &hostIntencities,
      const CameraModel &camTarget, const ResidualSettings &settings)
      : targetFrame(targetFrame)
      , hostFrameToBody(hostFrameToBody)
      , targetBodyToFrame(targetBodyToFrame)
      , dir(dir)
      , reprojPattern(reprojPattern)
      , hostIntencities(hostIntencities)
      , camTarget(camTarget)
      , settings(settings) {}

public:
  template <typename T>
  bool operator()(const T *hostToWorldQ, const T *hostToWorldT,
                  const T *targetToWorldQ, const T *targetToWorldT,
                  const T *worldToHostAff, const T *worldToTargetAff,
                  const T *logDepth, T *residual) const {
    using Vec2t = Eigen::Matrix<T, 2, 1>;
    using Vec3t = Eigen::Matrix<T, 3, 1>;
    using Mat33t = Eigen::Matrix<T, 3, 3>;
    using Quatt = Eigen::Quaternion<T>;
    using SE3t = Sophus::SE3<T>;
    using AffLightT = AffineLightTransform<T>;

    Eigen::Map<const Vec3t> hostToWorldTM(hostToWorldT);
    Eigen::Map<const Quatt> hostToWorldQM(hostToWorldQ);
    SE3t hostToWorld(hostToWorldQM, hostToWorldTM);
    Eigen::Map<const Vec3t> targetToWorldTM(targetToWorldT);
    Eigen::Map<const Quatt> targetToWorldQM(targetToWorldQ);
    SE3t targetToWorld(targetToWorldQM, targetToWorldTM);

    AffLightT lightWorldToHost(worldToHostAff[0], worldToHostAff[1]);
    AffLightT lightWorldToTarget(worldToTargetAff[0], worldToTargetAff[1]);

    Vec3t hostImVec = exp(*logDepth) * dir;
    Vec3t targetImVec =
        targetBodyToFrame * (targetToWorld.inverse() *
                             (hostToWorld * (hostFrameToBody * hostImVec)));
    Vec2t targetP = camTarget.map(targetImVec.data());
    for (int i = 0; i < hostIntencities.size(); ++i) {
      Vec2t patternP = targetP + reprojPattern[i].cast<T>();
      T targetIntensity;
      targetFrame->Evaluate(patternP[1], patternP[0], &targetIntensity);
      //      T transformedHostIntensity =
      //          lightWorldToTarget(lightWorldToHost.inverse()(T(hostIntencities[i])));
      T ea = exp(worldToTargetAff[0] - worldToHostAff[0]);
      T tmp = (T(hostIntencities[i]) - worldToHostAff[1]);
      T transformedHostIntensity = ea * tmp + worldToTargetAff[1];
      residual[i] = targetIntensity - transformedHostIntensity;
    }

    return true;
  }

private:
  PreKeyFrameEntryInternals::Interpolator_t *targetFrame;
  SE3 hostFrameToBody, targetBodyToFrame;
  Vec3 dir;
  static_vector<Vec2, Residual::MPS> reprojPattern;
  static_vector<double, Residual::MPS> hostIntencities;
  CameraModel camTarget;
  const ResidualSettings &settings;
};

struct EvalError {
  T valueErr;
  T hostToWorldRotErr;
  T hostToWorldTransErr;
  T targetToWorldRotErr;
  T targetToWorldTransErr;
  T lightWorldToHostErr;
  T lightWorldToTargetErr;
  T diffLogDepthErr;
};

TEST_F(ResidualTest, ValuesAndJacobian) {
  constexpr double transDrift = 0.02;
  constexpr double rotDrift = (M_PI / 180.0) * 0.004;
  const int patternSize = settings.residualPattern.pattern().size();

  SE3 f1ToF2GT = kf2->thisToWorld().inverse() * kf1->thisToWorld();
  double trans = f1ToF2GT.translation().norm();
  double transErr = trans * transDrift;
  double rotErr = trans * rotDrift;
  kf1->thisToWorld.setValue(kf1->thisToWorld() *
                            sampleSe3(rotErr, transErr, mt));
  kf2->thisToWorld.setValue(kf2->thisToWorld() *
                            sampleSe3(rotErr, transErr, mt));

  SE3 hostFrameToBody = cam1->bundle[0].thisToBody;
  SE3 targetBodyToFrame = cam2->bundle[0].bodyToThis;
  SE3 hostToWorld = kf1->thisToWorld();
  SE3 targetToWorld = kf2->thisToWorld();
  SE3 hostToTarget = targetBodyToFrame * targetToWorld.inverse() * hostToWorld *
                     hostFrameToBody;
  AffLightT lightWorldToHost = kf1->frames[0].lightWorldToThis.cast<T>();
  AffLightT lightWorldToTarget = kf2->frames[0].lightWorldToThis.cast<T>();
  AffLightT lightHostToTarget =
      (lightWorldToTarget * lightWorldToHost.inverse()).cast<T>();

  MotionDerivatives dHostToTarget(
      cam1->bundle[0].thisToBody.cast<T>(), kf1->thisToWorld().cast<T>(),
      kf2->thisToWorld().cast<T>(), cam2->bundle[0].bodyToThis.cast<T>());

  ceres::TrivialLoss loss;
  std::vector<EvalError> errors;
  for (OptimizedPoint &op : kf1->frames[0].optimizedPoints) {
    Residual residual(&cam1->bundle[0], &cam2->bundle[0], &kf1->frames[0],
                      &kf2->frames[0], &op, hostToTarget, &loss,
                      settings.getResidualSettings());
    auto actualValuesStatic =
        residual.getValues(hostToTarget, lightHostToTarget);
    VecX actualValues(patternSize);
    for (int i = 0; i < patternSize; ++i)
      actualValues[i] = actualValuesStatic[i];
    Residual::Jacobian jacobian =
        residual.getJacobian(hostToTarget.cast<T>(), dHostToTarget,
                             lightWorldToHost, lightHostToTarget);

    MatX4t actual_dr_dq_host(patternSize, 4);
    MatX3t actual_dr_dq_host_tang(patternSize, 3);
    MatX3t actual_dr_dt_host(patternSize, 3);
    MatX4t actual_dr_dq_target(patternSize, 4);
    MatX3t actual_dr_dq_target_tang(patternSize, 3);
    MatX3t actual_dr_dt_target(patternSize, 3);
    MatX2t actual_dr_daff_host(patternSize, 2);
    MatX2t actual_dr_daff_target(patternSize, 2);
    VecX actual_dr_dlogd(patternSize);
    for (int i = 0; i < patternSize; ++i) {
      actual_dr_dq_host.row(i) =
          jacobian.gradItarget[i].transpose() * jacobian.dhost.dp_dq;
      actual_dr_dt_host.row(i) =
          jacobian.gradItarget[i].transpose() * jacobian.dhost.dp_dt;
      actual_dr_dq_target.row(i) =
          jacobian.gradItarget[i].transpose() * jacobian.dtarget.dp_dq;
      actual_dr_dt_target.row(i) =
          jacobian.gradItarget[i].transpose() * jacobian.dtarget.dp_dt;
      actual_dr_daff_host.row(i) = jacobian.dhost.dr_dab[i].transpose();
      actual_dr_daff_target.row(i) = jacobian.dtarget.dr_dab[i].transpose();
      actual_dr_dlogd[i] =
          jacobian.gradItarget[i].transpose() * jacobian.dp_dlogd;
    }

    std::unique_ptr<ceres::CostFunction> residualCF(
        new ceres::AutoDiffCostFunction<ResidualCostFunctor, ceres::DYNAMIC, 4,
                                        3, 4, 3, 2, 2, 1>(
            new ResidualCostFunctor(
                &kf2->preKeyFrame->frames[0].internals->interpolator(0),
                hostFrameToBody, targetBodyToFrame, op.dir,
                residual.getReprojPattern(), residual.getHostIntensities(),
                cam2->bundle[0].cam, settings.getResidualSettings()),
            patternSize));

    const double *parameters[7] = {hostToWorld.so3().data(),
                                   hostToWorld.translation().data(),
                                   targetToWorld.so3().data(),
                                   targetToWorld.translation().data(),
                                   kf1->frames[0].lightWorldToThis.data,
                                   kf2->frames[0].lightWorldToThis.data,
                                   &op.logDepth};
    VecX expectedValues(patternSize);
    MatX4RM expected_dr_dq_host(patternSize, 4);
    MatX3t expected_dr_dq_host_tang(patternSize, 3);
    MatX3RM expected_dr_dt_host(patternSize, 3);
    MatX4RM expected_dr_dq_target(patternSize, 4);
    MatX3t expected_dr_dq_target_tang(patternSize, 3);
    MatX3RM expected_dr_dt_target(patternSize, 3);
    MatX2RM expected_dr_daff_host(patternSize, 2);
    MatX2RM expected_dr_daff_target(patternSize, 2);
    VecX expected_dr_dlogd(patternSize);
    double *expectedJacobians[7] = {
        expected_dr_dq_host.data(),   expected_dr_dt_host.data(),
        expected_dr_dq_target.data(), expected_dr_dt_target.data(),
        expected_dr_daff_host.data(), expected_dr_daff_target.data(),
        expected_dr_dlogd.data()};

    residualCF->Evaluate(parameters, expectedValues.data(), expectedJacobians);

    EvalError error;
    error.valueErr = (actualValues - expectedValues).norm();
    error.hostToWorldRotErr = tangentErr(
        expected_dr_dq_host.cast<T>().eval(),
        actual_dr_dq_host.cast<T>().eval(), hostToWorld.so3().cast<T>(),
        expected_dr_dq_host_tang, actual_dr_dq_host_tang);
    error.hostToWorldTransErr =
        (actual_dr_dt_host - expected_dr_dt_host.cast<T>()).norm();
    error.targetToWorldRotErr = tangentErr(
        expected_dr_dq_target.cast<T>().eval(),
        actual_dr_dq_target.cast<T>().eval(), targetToWorld.so3().cast<T>(),
        expected_dr_dq_target_tang, actual_dr_dq_target_tang);
    error.targetToWorldTransErr =
        (actual_dr_dt_target - expected_dr_dt_target.cast<T>()).norm();
    error.lightWorldToHostErr =
        (actual_dr_daff_host - expected_dr_daff_host.cast<T>()).norm();
    error.lightWorldToTargetErr =
        (actual_dr_daff_target - expected_dr_daff_target.cast<T>()).norm();

    ASSERT_LE(error.valueErr, valueEps)
        << "actual:\n"
        << actualValues.transpose() << "\nexpected:\n"
        << expectedValues.transpose() << "\n";
    ASSERT_LE(error.hostToWorldRotErr, diffRotEps)
        << "actual:\n"
        << actual_dr_dq_host_tang << "\nexpected:\n"
        << expected_dr_dq_host_tang << "\n";
    ASSERT_LE(error.hostToWorldTransErr, diffTransEps)
        << "actual:\n"
        << actual_dr_dt_host << "\nexpected:\n"
        << expected_dr_dt_host << "\n";
    ASSERT_LE(error.targetToWorldRotErr, diffRotEps)
        << "actual:\n"
        << actual_dr_dq_target_tang << "\nexpected:\n"
        << expected_dr_dq_target_tang << "\n";
    ASSERT_LE(error.targetToWorldTransErr, diffTransEps)
        << "actual:\n"
        << actual_dr_dt_target << "\nexpected:\n"
        << expected_dr_dt_target << "\n";
    ASSERT_LE(error.lightWorldToHostErr, diffAffEps)
        << "actual:\n"
        << actual_dr_daff_host << "\nexpected:\n"
        << expected_dr_daff_host << "\n";
    ASSERT_LE(error.lightWorldToTargetErr, diffAffEps)
        << "actual:\n"
        << actual_dr_daff_target << "\nexpected:\n"
        << expected_dr_daff_target << "\n";
    ASSERT_LE(error.diffLogDepthErr, diffDepthEps)
        << "actual:\n"
        << actual_dr_dlogd << "\nexpected:\n"
        << expected_dr_dlogd << "\n";

    errors.push_back(error);
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  return RUN_ALL_TESTS();
}
