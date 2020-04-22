#include "PreKeyFrameEntryInternals.h"
#include "data/MultiFovReader.h"
#include "optimize/Residual.h"
#include "system/CameraBundle.h"
#include "system/IdentityPreprocessor.h"
#include "util/util.h"
#include <ceres/ceres.h>
#include <chrono>
#include <gtest/gtest.h>
#include <system/FrameTracker.h>

using namespace mdso;
using namespace mdso::optimize;

using MatX2RM = Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>;
using MatX3RM = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
using MatX4RM = Eigen::Matrix<double, Eigen::Dynamic, 4, Eigen::RowMajor>;

using MatX3RMt = Eigen::Matrix<T, Eigen::Dynamic, 3, Eigen::RowMajor>;

using WeightsVector = VecRt;

DEFINE_string(mfov_dir, "/shared/datasets/mfov",
              "Root folder of the MultiFoV dataset.");

template <typename T> struct ErrorBounds;

template <> struct ErrorBounds<float> {
  static constexpr float resEps = 2;
  static constexpr float valueEps = 1e-2;
  static constexpr float diffRotEps = 2e-2;
  static constexpr float diffTransEps = 2e-2;
  static constexpr float diffAffEps = 1e-6;
  static constexpr float diffDepthEps = 1e-7;

  static constexpr float qtqtEps = 1e-5;
  static constexpr float qtabEps = 5e-5;
  static constexpr float ababEps = 1e-7;
  static constexpr float qtdEps = 1e-5;
  static constexpr float abdEps = 1e-4;
  static constexpr float ddEps = 1e-5;

  static constexpr float qtEps = 1e-4;
  static constexpr float abEps = 5e-5;
  static constexpr float dEps = 5e-5;

  static constexpr float minProximity = 1e-3;
};

template <> struct ErrorBounds<double> {
  static constexpr double resEps = 2;
  static constexpr double valueEps = 1e-10;
  static constexpr double diffRotEps = 1e-10;
  static constexpr double diffTransEps = 1e-10;
  static constexpr double diffAffEps = 1e-14;
  static constexpr double diffDepthEps = 1e-14;

  static constexpr double qtqtEps = 1e-10;
  static constexpr double qtabEps = 1e-10;
  static constexpr double ababEps = 1e-10;
  static constexpr double qtdEps = 1e-10;
  static constexpr double abdEps = 1e-10;
  static constexpr double ddEps = 1e-10;

  static constexpr double qtEps = 1e-10;
  static constexpr double abEps = 1e-10;
  static constexpr double dEps = 1e-12;

  static constexpr float minProximity = 1e-12;
};

template <typename T> T relOrAbs(T err, T nrm, T eps) {
  return abs(nrm) < eps ? err : err / nrm;
}

template <typename T, typename MatrixT>
T relMatrixErr(const MatrixT &actual, const MatrixT &expected, T eps) {
  return relOrAbs((actual - expected).norm(), expected.norm(), eps);
}

cv::Mat3b cvtAff(const cv::Mat3b &mat, const AffLight &aff) {
  cv::Mat3b result;
  cv::convertScaleAbs(mat, result, aff.ea(), aff.b());
  return result;
}

class ResidualTestBase : public ::testing::Test {
protected:
  static constexpr int fnum1 = 375, fnum2 = 385;

  void SetUp() override {
    residualSettings = settings.getResidualSettings();

    reader.reset(new MultiFovReader(FLAGS_mfov_dir));
    PixelSelector pixelSelectors[2];

    Timestamp ts1[] = {fnum1, fnum1};
    Timestamp ts2[] = {fnum2, fnum2};

    auto [kf1ImageToWorld, kf2ImageToWorld] = getKfImageToWorld();
    SE3 img1ToImg2 = kf2ImageToWorld.inverse() * kf1ImageToWorld;

    SE3 bodyToCam[] = {SE3::sampleUniform(mt), SE3::sampleUniform(mt)};
    //    SE3 bodyToCam[] = {SE3(), SE3()};
    bodyToCam[0].translation() *= img1ToImg2.translation().norm();
    bodyToCam[1].translation() *= img1ToImg2.translation().norm();
    CameraModel cameraModel = reader->cam().bundle[0].cam;
    CameraModel cams[] = {cameraModel, cameraModel};
    cam.reset(new CameraBundle(bodyToCam, cams, 2));

    SE3 kf1ToKf2 = bodyToCam[1].inverse() * img1ToImg2 * bodyToCam[0];

    IdentityPreprocessor idPrep;

    cv::Mat3b frame1 = reader->frame(fnum1)[0].frame,
              frame2 = reader->frame(fnum2)[0].frame;
    AffLight lightWorldToF1 = sampleAffLight<double>(settings.affineLight, mt);
    AffLight lightWorldToF2 = sampleAffLight<double>(settings.affineLight, mt);
    //        AffLight lightWorldToF1;
    //        AffLight lightWorldToF2;
    frame1 = lightWorldToF1(frame1);
    frame2 = lightWorldToF2(frame2);

    cv::Mat3b frames1Stub[] = {frame1.clone(), frame1.clone()};
    cv::Mat3b frames2Stub[] = {frame2.clone(), frame2.clone()};

    pixelSelectors[0].initialize(frame1, settings.keyFrame.immaturePointsNum());
    pixelSelectors[1].initialize(frame1, settings.keyFrame.immaturePointsNum());

    auto depths1 = reader->depths(fnum1);

    std::unique_ptr<PreKeyFrame> pkf1(new PreKeyFrame(
        nullptr, cam.get(), &idPrep, frames1Stub, 1, ts1, settings.pyramid));
    kf1.reset(new KeyFrame(std::move(pkf1), pixelSelectors, settings.keyFrame,
                           settings.getPointTracerSettings()));
    kf1->thisToWorld.setValue(kf1ImageToWorld * cam->bundle[0].bodyToThis);
    kf1->frames[0].lightWorldToThis = lightWorldToF1;
    kf1->frames[1].lightWorldToThis = lightWorldToF1;
    std::unique_ptr<PreKeyFrame> pkf2(new PreKeyFrame(
        kf1.get(), cam.get(), &idPrep, frames2Stub, 2, ts2, settings.pyramid));
    TrackingResult trackingResult(2);
    trackingResult.baseToTracked = kf1ToKf2;
    trackingResult.lightBaseToTracked[0] =
        trackingResult.lightBaseToTracked[1] =
            lightWorldToF2 * lightWorldToF1.inverse();
    pkf2->setTracked(trackingResult);
    kf2.reset(new KeyFrame(std::move(pkf2), pixelSelectors, settings.keyFrame,
                           settings.getPointTracerSettings()));

    for (ImmaturePoint &ip : kf1->frames[0].immaturePoints)
      if (ip.state == ImmaturePoint::ACTIVE) {
        auto maybeDepth1 = depths1->depth(0, ip.p);
        if (maybeDepth1) {
          ip.setTrueDepth(maybeDepth1.value(), settings.pointTracer);
          kf1->frames[0].optimizedPoints.emplace_back(ip);
        }
      }
    kf1->frames[0].immaturePoints.clear();

    SE3 hostFrameToBody = cam->bundle[0].thisToBody;
    SE3 targetBodyToFrame = cam->bundle[1].bodyToThis;
    SE3 hostToWorld = kf1->thisToWorld();
    SE3 targetToWorld = kf2->thisToWorld();
    hostToTarget = (targetBodyToFrame * targetToWorld.inverse() * hostToWorld *
                    hostFrameToBody)
                       .cast<T>();
    lightWorldToHost = kf1->frames[0].lightWorldToThis.cast<T>();
    lightWorldToTarget = kf2->frames[1].lightWorldToThis.cast<T>();
    lightHostToTarget =
        (lightWorldToTarget * lightWorldToHost.inverse()).cast<T>();

    dhostToTarget.reset(new MotionDerivatives(
        cam->bundle[0].thisToBody.cast<T>(), kf1->thisToWorld().cast<T>(),
        kf2->thisToWorld().cast<T>(), cam->bundle[1].bodyToThis.cast<T>()));

    lossFunction.reset(new ceres::TrivialLoss());
    auto &optimizedPoints = kf1->frames[0].optimizedPoints;
    logDepths.resize(optimizedPoints.size());
    residuals.reserve(optimizedPoints.size());
    for (int i = 0; i < optimizedPoints.size(); ++i) {
      OptimizedPoint &op = optimizedPoints[i];
      logDepths[i] = op.logDepth;
      residuals.emplace_back(0, 0, 1, 1, i, cam.get(), &kf1->frames[0],
                             &kf2->frames[1], &op, logDepths[i], hostToTarget,
                             lossFunction.get(), residualSettings);
    }
  }

  virtual std::pair<SE3, SE3> getKfImageToWorld() = 0;

  static constexpr T resEps = ErrorBounds<T>::resEps;
  static constexpr T valueEps = ErrorBounds<T>::valueEps;
  static constexpr T diffRotEps = ErrorBounds<T>::diffRotEps;
  static constexpr T diffTransEps = ErrorBounds<T>::diffTransEps;
  static constexpr T diffAffEps = ErrorBounds<T>::diffAffEps;
  static constexpr T diffDepthEps = ErrorBounds<T>::diffDepthEps;

  static constexpr T qtqtEps = ErrorBounds<T>::qtqtEps;
  static constexpr T qtabEps = ErrorBounds<T>::qtabEps;
  static constexpr T ababEps = ErrorBounds<T>::ababEps;
  static constexpr T qtdEps = ErrorBounds<T>::qtdEps;
  static constexpr T abdEps = ErrorBounds<T>::abdEps;
  static constexpr T ddEps = ErrorBounds<T>::ddEps;

  static constexpr T qtEps = ErrorBounds<T>::qtEps;
  static constexpr T abEps = ErrorBounds<T>::abEps;
  static constexpr T dEps = ErrorBounds<T>::dEps;

  static constexpr T minProximity = ErrorBounds<T>::minProximity;

  std::unique_ptr<MultiFovReader> reader;
  std::mt19937 mt;
  std::unique_ptr<CameraBundle> cam;
  std::unique_ptr<KeyFrame> kf1, kf2;
  std::unique_ptr<ceres::LossFunction> lossFunction;
  StdVector<Residual> residuals;
  std::vector<T> logDepths;
  Settings settings;
  ResidualSettings residualSettings;

  SE3t hostToTarget;
  std::unique_ptr<MotionDerivatives> dhostToTarget;
  AffLightT lightWorldToHost;
  AffLightT lightWorldToTarget;
  AffLightT lightHostToTarget;
};

class ResidualTestGtPoses : public ResidualTestBase {
private:
  std::pair<SE3, SE3> getKfImageToWorld() override {
    return {reader->frameToWorld(fnum1), reader->frameToWorld(fnum2)};
  }
};

class ResidualTestDriftedPoses : public ResidualTestBase {
private:
  static constexpr double transDrift = 0.02;
  static constexpr double rotDrift = (M_PI / 180.0) * 0.004;

  std::pair<SE3, SE3> getKfImageToWorld() override {
    auto kf1ToWorldGT = reader->frameToWorld(fnum1);
    auto kf2ToWorldGT = reader->frameToWorld(fnum2);
    SE3 f1ToF2GT = kf2ToWorldGT.inverse() * kf1ToWorldGT;
    double trans = f1ToF2GT.translation().norm();
    double transErr = trans * transDrift;
    double rotErr = trans * rotDrift;
    return {sampleSe3(rotErr, transErr, mt) * kf1ToWorldGT,
            sampleSe3(rotErr, transErr, mt) * kf2ToWorldGT};
  }
};

TEST_F(ResidualTestGtPoses, IsSmallOnGt) {
  static constexpr double expectedPercentileOfSmall = 0.1;

  StdVector<VecRt> resValues;

  for (int i = 0; i < residuals.size(); ++i) {
    OptimizedPoint &op = kf1->frames[0].optimizedPoints[i];
    resValues.push_back(
        residuals[i].getValues(hostToTarget, lightHostToTarget, op.logDepth));
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

double edgeProximity(double p0, double p1) {
  return std::min({p0 - std::floor(p0), std::ceil(p0) - p0, p1 - std::floor(p1),
                   std::ceil(p1) - p1});
}

template <typename T> double toDouble(const T &val) { return val.a; }
template <> double toDouble<double>(const double &val) { return val; }
template <> double toDouble<float>(const float &val) { return val; }

class ResidualCostFunctor {
public:
  ResidualCostFunctor(PreKeyFrameEntryInternals::Interpolator_t *targetFrame,
                      const SE3 &hostFrameToBody, const SE3 &targetBodyToFrame,
                      const Vec3 &dir,
                      const static_vector<Vec2t, Residual::MPS> &reprojPattern,
                      const VecRt &hostIntencities,
                      const CameraModel &camTarget,
                      const ResidualSettings &settings,
                      double *outLastProximity)
      : targetFrame(targetFrame)
      , hostFrameToBody(hostFrameToBody)
      , targetBodyToFrame(targetBodyToFrame)
      , dir(dir)
      , reprojPattern(reprojPattern)
      , hostIntencities(hostIntencities)
      , camTarget(camTarget)
      , settings(settings)
      , outLastProximity(outLastProximity) {}

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

    double lastProximity = INF;
    for (int i = 0; i < hostIntencities.size(); ++i) {
      Vec2t patternP = targetP + reprojPattern[i].cast<T>();
      lastProximity =
          std::min(lastProximity,
                   edgeProximity(toDouble(patternP[0]), toDouble(patternP[1])));
      T targetIntensity;
      targetFrame->Evaluate(patternP[1], patternP[0], &targetIntensity);
      //      T transformedHostIntensity =
      //          lightWorldToTarget(lightWorldToHost.inverse()(T(hostIntencities[i])));
      T ea = exp(worldToTargetAff[0] - worldToHostAff[0]);
      T tmp = (T(hostIntencities[i]) - worldToHostAff[1]);
      T transformedHostIntensity = ea * tmp + worldToTargetAff[1];
      residual[i] = targetIntensity - transformedHostIntensity;
    }

    *outLastProximity = lastProximity;

    return true;
  }

private:
  double *outLastProximity;
  PreKeyFrameEntryInternals::Interpolator_t *targetFrame;
  SE3 hostFrameToBody, targetBodyToFrame;
  Vec3 dir;
  static_vector<Vec2t, Residual::MPS> reprojPattern;
  VecRt hostIntencities;
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

TEST_F(ResidualTestDriftedPoses, AreValuesAndJacobianCorrect) {
  const int patternSize = settings.residualPattern.pattern().size();

  SE3 hostFrameToBody = cam->bundle[0].thisToBody;
  SE3 targetBodyToFrame = cam->bundle[1].bodyToThis;
  SE3 hostToWorld = kf1->thisToWorld();
  SE3 targetToWorld = kf2->thisToWorld();

  std::vector<EvalError> errors;
  double timeEvalMy = 0, timeEvalAuto = 0;
  int pointInd = 0;
  int numDiscarded = 0;
  for (const Residual &residual : residuals) {
    TimePoint start, end;
    start = now();
    double logDepth = logDepths[residual.pointInd()];
    Residual::CachedValues cachedValues(patternSize);
    auto actualValues = residual.getValues(hostToTarget, lightHostToTarget,
                                           logDepth, &cachedValues);
    Residual::Jacobian jacobian =
        residual.getJacobian(hostToTarget, *dhostToTarget, lightWorldToHost,
                             lightHostToTarget, logDepth, cachedValues);

    end = now();
    timeEvalMy += secondsBetween(start, end);

    auto actual_dr_dq_host = jacobian.dr_dq_host();
    MatR3t actual_dr_dq_host_tang(patternSize, 3);
    auto actual_dr_dt_host = jacobian.dr_dt_host();
    auto actual_dr_dq_target = jacobian.dr_dq_target();
    MatR3t actual_dr_dq_target_tang(patternSize, 3);
    auto actual_dr_dt_target = jacobian.dr_dt_target();
    auto actual_dr_daff_host = jacobian.dr_daff_host();
    auto actual_dr_daff_target = jacobian.dr_daff_target();
    auto actual_dr_dlogd = jacobian.dr_dlogd();

    double proximity = 0;
    ResidualCostFunctor *residualCostFunctor = new ResidualCostFunctor(
        &kf2->preKeyFrame->frames[1].internals->interpolator(0),
        hostFrameToBody, targetBodyToFrame,
        residual.getHostDir().cast<double>(), residual.getReprojPattern(),
        residual.getHostIntensities(), cam->bundle[1].cam, residualSettings,
        &proximity);

    ceres::AutoDiffCostFunction<ResidualCostFunctor, ceres::DYNAMIC, 4, 3, 4, 3,
                                2, 2, 1>
        residualCF(residualCostFunctor, patternSize);

    const double *parameters[7] = {hostToWorld.so3().data(),
                                   hostToWorld.translation().data(),
                                   targetToWorld.so3().data(),
                                   targetToWorld.translation().data(),
                                   kf1->frames[0].lightWorldToThis.data,
                                   kf2->frames[1].lightWorldToThis.data,
                                   &logDepth};
    VecX expectedValues(patternSize);
    MatX4RM expected_dr_dq_host(patternSize, 4);
    MatX3RMt expected_dr_dq_host_tang(patternSize, 3);
    MatX3RM expected_dr_dt_host(patternSize, 3);
    MatX4RM expected_dr_dq_target(patternSize, 4);
    MatX3RMt expected_dr_dq_target_tang(patternSize, 3);
    MatX3RM expected_dr_dt_target(patternSize, 3);
    MatX2RM expected_dr_daff_host(patternSize, 2);
    MatX2RM expected_dr_daff_target(patternSize, 2);
    VecX expected_dr_dlogd(patternSize);
    double *expectedJacobians[7] = {
        expected_dr_dq_host.data(),   expected_dr_dt_host.data(),
        expected_dr_dq_target.data(), expected_dr_dt_target.data(),
        expected_dr_daff_host.data(), expected_dr_daff_target.data(),
        expected_dr_dlogd.data()};

    start = std::chrono::system_clock::now();

    residualCF.Evaluate(parameters, expectedValues.data(), expectedJacobians);

    end = std::chrono::system_clock::now();

    timeEvalAuto += secondsBetween(start, end);

    if (proximity < minProximity) {
      numDiscarded++;
      continue;
    }

    EvalError error;
    error.valueErr = (actualValues - expectedValues.cast<T>()).norm();
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

    EXPECT_LE(relOrAbs(error.valueErr, T(expectedValues.norm()), valueEps),
              valueEps)
        << "actual:\n"
        << actualValues.transpose() << "\nexpected:\n"
        << expectedValues.transpose() << "\n";
    EXPECT_LE(relOrAbs(error.hostToWorldRotErr, expected_dr_dq_host_tang.norm(),
                       diffRotEps),
              diffRotEps)
        << "actual:\n"
        << actual_dr_dq_host_tang << "\nexpected:\n"
        << expected_dr_dq_host_tang
        << "\nreproj = " << cachedValues.reproj.transpose()
        << "\nproximity = " << proximity;
    EXPECT_LE(relOrAbs(error.hostToWorldTransErr, T(expected_dr_dt_host.norm()),
                       diffTransEps),
              diffTransEps)
        << "actual:\n"
        << actual_dr_dt_host << "\nexpected:\n"
        << expected_dr_dt_host << "\n";
    EXPECT_LE(relOrAbs(error.targetToWorldRotErr,
                       expected_dr_dq_target_tang.norm(), diffRotEps),
              diffRotEps)
        << "actual:\n"
        << actual_dr_dq_target_tang << "\nexpected:\n"
        << expected_dr_dq_target_tang << "\n";
    EXPECT_LE(relOrAbs(error.targetToWorldTransErr,
                       T(expected_dr_dt_target.norm()), diffTransEps),
              diffTransEps)
        << "actual:\n"
        << actual_dr_dt_target << "\nexpected:\n"
        << expected_dr_dt_target
        << "\nabs error = " << error.targetToWorldTransErr << "\n";
    EXPECT_LE(relOrAbs(error.lightWorldToHostErr,
                       T(expected_dr_daff_host.norm()), diffAffEps),
              diffAffEps)
        << "actual:\n"
        << actual_dr_daff_host << "\nexpected:\n"
        << expected_dr_daff_host << "\n";
    EXPECT_LE(relOrAbs(error.lightWorldToTargetErr,
                       T(expected_dr_daff_target.norm()), diffAffEps),
              diffAffEps)
        << "actual:\n"
        << actual_dr_daff_target << "\nexpected:\n"
        << expected_dr_daff_target << "\n";
    EXPECT_LE(relOrAbs(error.diffLogDepthErr, T(expected_dr_dlogd.norm()),
                       diffDepthEps),
              diffDepthEps)
        << "actual:\n"
        << actual_dr_dlogd << "\nexpected:\n"
        << expected_dr_dlogd << "\n";

    errors.push_back(error);
  }

  double relDiscarded = double(numDiscarded) / residuals.size();

  EXPECT_LT(relDiscarded, 0.05) << "Too much points were discarded";

  LOG(INFO) << "discarded because of the edge proximity: " << numDiscarded
            << " of " << residuals.size();
  LOG(INFO) << "total time on smart jacobian eval: " << timeEvalMy << std::endl;
  LOG(INFO) << "total time on auto  jacobian eval: " << timeEvalAuto
            << std::endl;
}

Residual::FrameFrameHessian
getFrameFrameHessian(const MatR4t &dr_dq_frame1, const MatR3t &dr_dt_frame1,
                     const MatR2t &dr_daff_frame1, const MatR4t &dr_dq_frame2,
                     const MatR3t &dr_dt_frame2, const MatR2t &dr_daff_frame2,
                     const WeightsVector &weights) {
  Residual::FrameFrameHessian result;

  MatR7t dr_dqt_frame1(dr_dq_frame1.rows(), 7);
  dr_dqt_frame1 << dr_dq_frame1, dr_dt_frame1;
  MatR7t dr_dqt_frame2(dr_dq_frame2.rows(), 7);
  dr_dqt_frame2 << dr_dq_frame2, dr_dt_frame2;
  Eigen::Map<const VecRt> wMap(weights.data(), weights.size());
  Eigen::DiagonalMatrix<T, Eigen::Dynamic, Residual::MPS> w = wMap.asDiagonal();

  result.qtqt = dr_dqt_frame1.transpose() * w * dr_dqt_frame2;
  result.qtab = dr_dqt_frame1.transpose() * w * dr_daff_frame2;
  result.abqt = dr_daff_frame1.transpose() * w * dr_dqt_frame2;
  result.abab = dr_daff_frame1.transpose() * w * dr_daff_frame2;

  return result;
}

Residual::FramePointHessian getFramePointHessian(const MatR4t &dr_dq,
                                                 const MatR3t &dr_dt,
                                                 const MatR2t &dr_daff,
                                                 const VecRt &dr_dlogd,
                                                 const WeightsVector &weights) {
  Residual::FramePointHessian result;
  MatR7t dr_dqt(dr_dq.rows(), 7);
  dr_dqt << dr_dq, dr_dt;
  Eigen::Map<const VecRt> wMap(weights.data(), weights.size());
  Eigen::DiagonalMatrix<T, Eigen::Dynamic, Residual::MPS> w = wMap.asDiagonal();
  result.qtd = dr_dqt.transpose() * w * dr_dlogd;
  result.abd = dr_daff.transpose() * w * dr_dlogd;

  return result;
}

T getPointPointHessian(const VecRt &dr_dlogd, const WeightsVector &weights) {
  CHECK_EQ(dr_dlogd.size(), weights.size());

  T result = 0;
  for (int i = 0; i < weights.size(); ++i)
    result += weights[i] * dr_dlogd[i] * dr_dlogd[i];
  return result;
}

Residual::DeltaHessian
getExpectedDeltaHessian(const Residual::Jacobian &jacobian,
                        const WeightsVector &weights) {
  int PS = weights.size();
  Residual::DeltaHessian deltaHessian;
  MatR4t dr_dq_host = jacobian.dr_dq_host();
  MatR3t dr_dt_host = jacobian.dr_dt_host();
  MatR2t dr_daff_host = jacobian.dr_daff_host();
  MatR4t dr_dq_target = jacobian.dr_dq_target();
  MatR3t dr_dt_target = jacobian.dr_dt_target();
  MatR2t dr_daff_target = jacobian.dr_daff_target();
  VecRt dr_dlogd = jacobian.dr_dlogd();

  deltaHessian.hostHost =
      getFrameFrameHessian(dr_dq_host, dr_dt_host, dr_daff_host, dr_dq_host,
                           dr_dt_host, dr_daff_host, weights);
  deltaHessian.hostTarget =
      getFrameFrameHessian(dr_dq_host, dr_dt_host, dr_daff_host, dr_dq_target,
                           dr_dt_target, dr_daff_target, weights);
  deltaHessian.targetTarget =
      getFrameFrameHessian(dr_dq_target, dr_dt_target, dr_daff_target,
                           dr_dq_target, dr_dt_target, dr_daff_target, weights);
  deltaHessian.hostPoint = getFramePointHessian(
      dr_dq_host, dr_dt_host, dr_daff_host, dr_dlogd, weights);
  deltaHessian.targetPoint = getFramePointHessian(
      dr_dq_target, dr_dt_target, dr_daff_target, dr_dlogd, weights);
  deltaHessian.pointPoint = getPointPointHessian(dr_dlogd, weights);

  return deltaHessian;
}

TEST_F(ResidualTestDriftedPoses, IsDeltaHessianCorrect) {
  const int patternSize = settings.residualPattern.pattern().size();

  for (const Residual &residual : residuals) {
    Residual::CachedValues cachedValues(patternSize);
    T logDepth = logDepths[residual.pointInd()];
    VecRt values = residual.getValues(hostToTarget, lightHostToTarget, logDepth,
                                      &cachedValues);
    VecRt weights = residual.getHessianWeights(values);
    Residual::Jacobian jacobian =
        residual.getJacobian(hostToTarget, *dhostToTarget, lightWorldToHost,
                             lightHostToTarget, logDepth, cachedValues);
    Residual::DeltaHessian actual = residual.getDeltaHessian(values, jacobian);
    Residual::DeltaHessian expected =
        getExpectedDeltaHessian(jacobian, weights);

    //    T err = relMatrixErr(actual.hostHost.qtqt, expected.hostHost.qtqt,
    //    0.1); EXPECT_LE(err,
    //              0.1);
    //    errors.push_back(err);
    EXPECT_LE(
        relMatrixErr(actual.hostHost.qtqt, expected.hostHost.qtqt, qtqtEps),
        qtqtEps);
    EXPECT_LE(
        relMatrixErr(actual.hostHost.qtab, expected.hostHost.qtab, qtqtEps),
        qtabEps);
    EXPECT_LE(
        relMatrixErr(actual.hostHost.abqt, expected.hostHost.abqt, qtabEps),
        qtabEps);
    EXPECT_LE(
        relMatrixErr(actual.hostHost.abab, expected.hostHost.abab, ababEps),
        ababEps);

    EXPECT_LE(
        relMatrixErr(actual.hostTarget.qtqt, expected.hostTarget.qtqt, qtqtEps),
        qtqtEps);
    EXPECT_LE(
        relMatrixErr(actual.hostTarget.qtab, expected.hostTarget.qtab, qtqtEps),
        qtabEps);
    EXPECT_LE(
        relMatrixErr(actual.hostTarget.abqt, expected.hostTarget.abqt, qtabEps),
        qtabEps);
    EXPECT_LE(
        relMatrixErr(actual.hostTarget.abab, expected.hostTarget.abab, ababEps),
        ababEps);

    EXPECT_LE(relMatrixErr(actual.targetTarget.qtqt, expected.targetTarget.qtqt,
                           qtqtEps),
              qtqtEps);
    EXPECT_LE(relMatrixErr(actual.targetTarget.qtab, expected.targetTarget.qtab,
                           qtqtEps),
              qtabEps);
    EXPECT_LE(relMatrixErr(actual.targetTarget.abqt, expected.targetTarget.abqt,
                           qtabEps),
              qtabEps);
    EXPECT_LE(relMatrixErr(actual.targetTarget.qtqt, expected.targetTarget.qtqt,
                           qtqtEps),
              qtqtEps);

    EXPECT_LE(
        relMatrixErr(actual.hostPoint.qtd, expected.hostPoint.qtd, qtdEps),
        qtdEps);
    EXPECT_LE(
        relMatrixErr(actual.hostPoint.abd, expected.hostPoint.abd, abdEps),
        abdEps);

    EXPECT_LE(
        relMatrixErr(actual.targetPoint.qtd, expected.targetPoint.qtd, qtdEps),
        qtdEps);
    EXPECT_LE(
        relMatrixErr(actual.targetPoint.abd, expected.targetPoint.abd, abdEps),
        abdEps);

    EXPECT_LE(relOrAbs(actual.pointPoint - expected.pointPoint,
                       expected.pointPoint, ddEps),
              ddEps);
  }
}

Residual::FrameGradient getExpectedFrameGradient(const MatR4t &dr_dq,
                                                 const MatR3t &dr_dt,
                                                 const MatR2t &dr_dab,
                                                 const VecRt &values,
                                                 const VecRt &weights) {
  int PS = values.size();
  MatR7t dr_dqt(PS, 7);
  dr_dqt << dr_dq, dr_dt;
  Residual::FrameGradient frameGradient;
  frameGradient.qt = dr_dqt.transpose() * weights.asDiagonal() * values;
  frameGradient.ab = dr_dab.transpose() * weights.asDiagonal() * values;
  return frameGradient;
}

Residual::DeltaGradient
getExpectedDeltaGradient(const Residual::Jacobian &jacobian,
                         const VecRt &values, const VecRt &weights) {
  int PS = values.size();
  Residual::DeltaGradient deltaGradient;
  deltaGradient.host =
      getExpectedFrameGradient(jacobian.dr_dq_host(), jacobian.dr_dt_host(),
                               jacobian.dr_daff_host(), values, weights);
  deltaGradient.target =
      getExpectedFrameGradient(jacobian.dr_dq_target(), jacobian.dr_dt_target(),
                               jacobian.dr_daff_target(), values, weights);
  deltaGradient.point =
      jacobian.dr_dlogd().transpose() * weights.asDiagonal() * values;
  return deltaGradient;
}

TEST_F(ResidualTestDriftedPoses, IsDeltaGradientCorrect) {
  for (const Residual &residual : residuals) {
    T logDepth = logDepths[residual.pointInd()];
    Residual::CachedValues cachedValues(residualSettings.patternSize());
    VecRt values = residual.getValues(hostToTarget, lightHostToTarget, logDepth,
                                      &cachedValues);
    VecRt weights = residual.getGradientWeights(values);
    Residual::Jacobian jacobian =
        residual.getJacobian(hostToTarget, *dhostToTarget, lightWorldToHost,
                             lightHostToTarget, logDepth, cachedValues);
    Residual::DeltaGradient actual =
        residual.getDeltaGradient(values, jacobian);
    Residual::DeltaGradient expected =
        getExpectedDeltaGradient(jacobian, values, weights);

    EXPECT_LE(relMatrixErr(actual.host.qt, expected.host.qt, qtEps), qtEps);
    EXPECT_LE(relMatrixErr(actual.host.ab, expected.host.ab, abEps), abEps);
    EXPECT_LE(relMatrixErr(actual.target.qt, expected.target.qt, qtEps), qtEps);
    EXPECT_LE(relMatrixErr(actual.target.ab, expected.target.ab, abEps), abEps);
    EXPECT_LE(
        relOrAbs(std::abs(actual.point - expected.point), expected.point, dEps),
        dEps);
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  return RUN_ALL_TESTS();
}
