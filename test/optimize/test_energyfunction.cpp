#include "data/MultiFovReader.h"
#include "internal/optimize/EnergyFunctionCeres.h"
#include "optimize/EnergyFunction.h"
#include "system/BundleAdjusterCeres.h"
#include "system/IdentityPreprocessor.h"
#include "util/flags.h"
#include <gtest/gtest.h>

DEFINE_string(mfov_dir, "/shared/datasets/mfov",
              "Root folder of the MultiFoV dataset.");

DEFINE_int32(opt_count, 100, "Number of optimized points in the problem.");
DEFINE_bool(profile_only, false,
            "If set to true, no checks are performed, nor are expected hessian "
            "or gradient computed.");
DEFINE_int32(num_evaluations, 1000,
             "Number of evaluations to perform for profiling (is only used "
             "when --profile_only is on).");

using namespace mdso;
using namespace mdso::optimize;

template <typename T> struct ErrorBounds;

template <> struct ErrorBounds<float> {
  static constexpr float hessianRelErr = 1e-2;
  static constexpr float gradientRelErr = 2e-2;
  static constexpr float predictionRelErr = 1e-2;
  static constexpr float energyRelErr = 1e-4;
  static constexpr float jacobianRelErr = 1e-4;
};

template <> struct ErrorBounds<double> {
  static constexpr double hessianRelErr = 1e-10;
  static constexpr double gradientRelErr = 1e-10;
  static constexpr double predictionRelErr = 1e-10;
  static constexpr float energyRelErr = 1e-12;
  static constexpr float jacobianRelErr = 1e-4;
};

class EnergyFunctionTest : public ::testing::Test {
protected:
  static constexpr int keyFramesCount = 7;
  static constexpr int keyFrameNums[keyFramesCount] = {375, 380, 384, 390,
                                                       398, 401, 407};

  static constexpr int sndDoF = SO3xS2Parametrization::DoF,
                       restDoF = RightExpParametrization<SE3t>::DoF;
  static constexpr int affDoF = AffLightT::DoF;
  static constexpr int sndFrameDoF = sndDoF + affDoF,
                       restFrameDoF = restDoF + affDoF;

  static constexpr double transDrift = 0.002;
  static constexpr double rotDrift = (M_PI / 180.0) * 0.004;

  EnergyFunctionTest()
      : pointsPerFrame(FLAGS_opt_count / keyFramesCount) {}

  void SetUp() override {
    settings = getFlaggedSettings();
    settings.setMaxOptimizedPoints(pointsPerFrame * keyFramesCount);
    settings.keyFrame.setImmaturePointsNum(pointsPerFrame);
    energyFunctionSettings = settings.getEnergyFunctionSettings();

    reader.reset(new MultiFovReader(FLAGS_mfov_dir));
    SE3 bodyToFrame = SE3::sampleUniform(mt);
    bodyToFrame.translation() *= 0.05;
    CameraModel cameraModel = reader->cam().bundle[0].cam;
    cam.reset(new CameraBundle(&bodyToFrame, &cameraModel, 1));

    frameParameterOrder.reset(
        new FrameParameterOrder(keyFramesCount, cam->bundle.size()));

    PixelSelector pixelSelector;
    pixelSelector.initialize(reader->frame(keyFrameNums[0])[0].frame,
                             settings.keyFrame.immaturePointsNum());
    keyFrames.reserve(keyFramesCount);
    SE3 worldToFirstGT = reader->frameToWorld(keyFrameNums[0]).inverse();
    for (int i = 0; i < keyFramesCount; ++i) {
      Timestamp ts = keyFrameNums[i];
      AffLight lightWorldToF = sampleAffLight<double>(settings.affineLight, mt);
      cv::Mat3b coloredFrame = reader->frame(keyFrameNums[i])[0].frame;
      coloredFrame = lightWorldToF(coloredFrame);
      std::unique_ptr<PreKeyFrame> pkf(
          new PreKeyFrame(nullptr, cam.get(), &idPrep, &coloredFrame, i, &ts,
                          settings.pyramid));
      keyFrames.emplace_back(new KeyFrame(std::move(pkf), &pixelSelector,
                                          settings.keyFrame,
                                          settings.getPointTracerSettings()));
      keyFrames.back()->frames[0].lightWorldToThis = lightWorldToF;

      SE3 frameToWorldGT = reader->frameToWorld(keyFrameNums[i]);
      SE3 thisToFirstGT = worldToFirstGT * frameToWorldGT;
      double transErr = thisToFirstGT.translation().norm() * transDrift;
      double rotErr = thisToFirstGT.so3().log().norm() * rotDrift;
      SE3 frameToWorld = i == 0
                             ? frameToWorldGT
                             : frameToWorldGT * sampleSe3(rotErr, transErr, mt);

      keyFrames.back()->thisToWorld.setValue(frameToWorld *
                                             cam->bundle[0].bodyToThis);

      auto depths = reader->depths(keyFrameNums[i]);
      for (ImmaturePoint &ip : keyFrames.back()->frames[0].immaturePoints) {
        auto maybeDepth = depths->depth(0, ip.p);
        if (maybeDepth) {
          ip.setTrueDepth(maybeDepth.value(), settings.pointTracer);
          keyFrames.back()->frames[0].optimizedPoints.emplace_back(ip);
        }
      }
      keyFrames.back()->frames[0].immaturePoints.clear();
    }

    secondFrameParam.reset(new SO3xS2Parametrization(
        keyFrames[0]->thisToWorld(), keyFrames[1]->thisToWorld()));
    restFrameParams.reserve(keyFramesCount - 2);
    for (int i = 2; i < keyFramesCount; ++i)
      restFrameParams.emplace_back(keyFrames[i]->thisToWorld().cast<T>());

    std::vector<KeyFrame *> kfPtrs;
    kfPtrs.reserve(keyFrames.size());
    for (auto &kf : keyFrames)
      kfPtrs.push_back(kf.get());
    energyFunction.reset(new EnergyFunction(
        cam.get(), kfPtrs.data(), kfPtrs.size(), energyFunctionSettings));
  }

  MatXXt getExpectedJacobian() {
    int PS = settings.residualPattern.pattern().size();
    MatXXt expectedJacobian =
        MatXXt::Zero(PS * energyFunction->getResiduals().size(),
                     frameParameterOrder->totalFrameParameters() +
                         energyFunction->numPoints());
    std::cout << "evaluating jacobian: 0% ... ";
    std::cout.flush();
    int percent = 0, step = 10;
    int numResiduals = energyFunction->getResiduals().size();
    for (int ri = 0; ri < numResiduals; ++ri) {
      const Residual &res = energyFunction->getResiduals()[ri];
      int hi = res.hostInd(), hci = res.hostCamInd(), ti = res.targetInd(),
          tci = res.targetCamInd(), pi = res.pointInd();

      KeyFrame *host = keyFrames[hi].get();
      KeyFrame *target = keyFrames[ti].get();
      SE3t hostToWorld = host->thisToWorld().cast<T>();
      SE3t targetToWorld = target->thisToWorld().cast<T>();
      SE3t hostFrameToBody = cam->bundle[0].thisToBody.cast<T>();
      SE3t targetBodyToFrame = cam->bundle[0].bodyToThis.cast<T>();

      AffLightT lightHostToTarget = (target->frames[0].lightWorldToThis *
                                     host->frames[0].lightWorldToThis.inverse())
                                        .cast<T>();

      MotionDerivatives dHostToTarget(hostFrameToBody, hostToWorld,
                                      targetToWorld, targetBodyToFrame);
      SE3t hostToTarget = targetBodyToFrame * targetToWorld.inverse() *
                          hostToWorld * hostFrameToBody;

      Residual::CachedValues cachedValues(PS);
      VecRt values =
          res.getValues(hostToTarget, lightHostToTarget,
                        energyFunction->getLogDepth(res), &cachedValues);

      Residual::Jacobian rj = res.getJacobian(
          hostToTarget, dHostToTarget,
          host->frames[0].lightWorldToThis.cast<T>(), lightHostToTarget,
          energyFunction->getLogDepth(res), cachedValues);

      setFrame(expectedJacobian, ri, hi, hci, rj.dr_dq_host(), rj.dr_dt_host(),
               rj.dr_daff_host());
      setFrame(expectedJacobian, ri, ti, tci, rj.dr_dq_target(),
               rj.dr_dt_target(), rj.dr_daff_target());
      int depthCol = frameParameterOrder->totalFrameParameters() + pi;
      expectedJacobian.block(ri * PS, depthCol, PS, 1) = rj.dr_dlogd();

      int oldPercent = percent;
      percent = double(ri + 1) / numResiduals * 100.0;
      if (percent == 100)
        std::cout << "100%" << std::endl;
      else if (oldPercent != percent && percent % step == 0) {
        std::cout << percent << "% ... ";
        std::cout.flush();
      }
    }

    return expectedJacobian;
  }

  double transError() const {
    auto kf0ToWorldGT = reader->frameToWorld(keyFrameNums[0]);
    double sumErr = 0;
    for (int i = 1; i < keyFramesCount; ++i) {
      auto kfiToWorldGT = reader->frameToWorld(keyFrameNums[i]);
      SE3 err = keyFrames[i]->thisToWorld() * cam->bundle[0].thisToBody *
                kfiToWorldGT.inverse();
      SE3 thisToFirstGT = kf0ToWorldGT.inverse() * kfiToWorldGT;
      //      sumErr += err.translation().norm() /
      //      thisToFirstGT.translation().norm();
      sumErr += err.translation().norm();
    }

    return 100 * sumErr / (keyFramesCount - 1);
  }

  double rotError() const {
    auto kf0ToWorldGT = reader->frameToWorld(keyFrameNums[0]);
    double sumErr = 0;
    for (int i = 0; i < keyFramesCount; ++i) {
      auto kfiToWorldGT = reader->frameToWorld(keyFrameNums[i]);
      SE3 err = keyFrames[i]->thisToWorld() * cam->bundle[0].thisToBody *
                kfiToWorldGT.inverse();
      SE3 thisToFirstGT = kf0ToWorldGT.inverse() * kfiToWorldGT;
      //      sumErr += err.so3().log().norm() /
      //      thisToFirstGT.translation().norm();
      sumErr += err.so3().log().norm();
    }
    return sumErr * (180. / M_PI) / (keyFramesCount - 1);
  }

  std::unique_ptr<MultiFovReader> reader;
  std::unique_ptr<CameraBundle> cam;
  std::unique_ptr<FrameParameterOrder> frameParameterOrder;
  std::vector<std::unique_ptr<KeyFrame>> keyFrames;
  std::unique_ptr<EnergyFunction> energyFunction;
  std::unique_ptr<SecondFrameParametrization> secondFrameParam;
  std::vector<FrameParametrization> restFrameParams;
  Settings settings;
  EnergyFunctionSettings energyFunctionSettings;
  IdentityPreprocessor idPrep;
  std::mt19937 mt;
  int pointsPerFrame;

private:
  void setFrame(MatXXt &jacobian, int residualInd, int frameInd, int camInd,
                const MatR4t &dr_dq, const MatR3t &dr_dt,
                const MatR2t &dr_daff) {
    if (frameInd == 0)
      return;
    if (frameInd == 1)
      setSecondFrame(jacobian, residualInd, camInd, dr_dq, dr_dt, dr_daff);
    else
      setOtherFrame(jacobian, residualInd, frameInd, camInd, dr_dq, dr_dt,
                    dr_daff);
  }

  void setSecondFrame(MatXXt &jacobian, int residualInd, int camInd,
                      const MatR4t &dr_dq, const MatR3t &dr_dt,
                      const MatR2t &dr_daff) {
    int PS = settings.residualPattern.pattern().size();
    int startRow = residualInd * PS;
    MatR7t dr_dqt(PS, 7);
    dr_dqt << dr_dq, dr_dt;
    Mat75t diffPlus = secondFrameParam->diffPlus();
    MatR5t dr_dqt_tang = dr_dqt * diffPlus;
    jacobian.block(startRow, frameParameterOrder->frameToWorld(1), PS, sndDoF) =
        dr_dqt_tang;
    jacobian.block(startRow, frameParameterOrder->lightWorldToFrame(1, camInd),
                   PS, 2) = dr_daff;
  }

  void setOtherFrame(MatXXt &jacobian, int residualInd, int frameInd,
                     int camInd, const MatR4t &dr_dq, const MatR3t &dr_dt,
                     const MatR2t &dr_daff) {
    CHECK_GE(frameInd, 2);
    int PS = settings.residualPattern.pattern().size();
    int startRow = residualInd * PS;
    MatR7t dr_dqt(PS, 7);
    dr_dqt << dr_dq, dr_dt;
    Mat76t diffPlus = restFrameParams[frameInd - 2].diffPlus();
    MatR6t dr_dqt_tang = dr_dqt * diffPlus;

    jacobian.block(startRow, frameParameterOrder->frameToWorld(frameInd), PS,
                   restDoF) = dr_dqt_tang;
    jacobian.block(startRow,
                   frameParameterOrder->lightWorldToFrame(frameInd, camInd), PS,
                   2) = dr_daff;
  }
};

double fillFactor(const MatXX &mat, double eps) {
  return double((mat.array().abs() >= eps).count()) / mat.size();
}

template <typename MatrixT> int countNaNs(const MatrixT &mat) {
  return (mat.array() != mat.array()).count();
}

TEST_F(EnergyFunctionTest, isEnergyCorrect) {
  std::vector<KeyFrame *> kfPtrs;
  kfPtrs.reserve(keyFrames.size());
  for (auto &kf : keyFrames)
    kfPtrs.push_back(kf.get());

  EnergyFunctionCeres energyFunctionCeres(kfPtrs.data(), kfPtrs.size(),
                                          settings.getBundleAdjusterSettings());
  ceres::Problem &problem = energyFunctionCeres.problem();
  ASSERT_EQ(problem.NumResidualBlocks(),
            energyFunction->getResiduals().size() *
                settings.residualPattern.pattern().size());

  double myEnergy = energyFunction->totalEnergy();
  double ceresEnergy;
  problem.Evaluate(ceres::Problem::EvaluateOptions(), &ceresEnergy, nullptr,
                   nullptr, nullptr);
  EXPECT_NEAR(myEnergy / 2, ceresEnergy,
              ceresEnergy * ErrorBounds<T>::energyRelErr);
}

DeltaParameterVector sampleDelta(const FrameParameterOrder &frameParameterOrder,
                                 int numPoints, double scale = 1) {
  DeltaParameterVector delta(frameParameterOrder.numKeyFrames(),
                             frameParameterOrder.numCameras(), numPoints);

  auto deltaF = delta.getFrame();
  auto deltaP = delta.getPoint();
  std::mt19937 mt;
  std::uniform_real_distribution<double> d(-scale, scale);
  for (int i = 0; i < deltaF.size(); ++i)
    deltaF[i] = d(mt);
  for (int i = 0; i < deltaP.size(); ++i)
    deltaP[i] = d(mt);
  delta =
      DeltaParameterVector(frameParameterOrder.numKeyFrames(),
                           frameParameterOrder.numCameras(), deltaF, deltaP);
  return delta;
}

VecXt getResidualValues(EnergyFunction &energyFunction, int patternSize) {
  const auto &residuals = energyFunction.getResiduals();
  VecXt values(residuals.size() * patternSize);
  for (int ri = 0; ri < residuals.size(); ++ri)
    values.segment(ri * patternSize, patternSize) =
        energyFunction.getResidualValues(ri);
  return values;
}

VecXt getJacobianDelta(EnergyFunction &energyFunction,
                       const DeltaParameterVector &delta, int patternSize) {
  const auto &residuals = energyFunction.getResiduals();
  VecXt values(residuals.size() * patternSize);
  for (int ri = 0; ri < residuals.size(); ++ri)
    values.segment(ri * patternSize, patternSize) =
        energyFunction.getPredictedResidualIncrement(ri, delta);
  return values;
}

TEST_F(EnergyFunctionTest, areDerivativesCorrect) {
  constexpr double deltaScale = 1e-10;
  int patternSize = settings.residualPattern.pattern().size();
  std::shared_ptr<Parameters> parameters = energyFunction->getParameters();
  Parameters::State state = parameters->saveState();
  DeltaParameterVector delta = sampleDelta(
      *frameParameterOrder, energyFunction->numPoints(), deltaScale);

  VecXt framed = delta.getFrame(), pointd = delta.getPoint();
  //  framed.setZero();
  //  pointd.setZero();
  delta =
      DeltaParameterVector(frameParameterOrder->numKeyFrames(),
                           frameParameterOrder->numCameras(), framed, pointd);

  VecXt twoJDeltaX = 2 * getJacobianDelta(*energyFunction, delta, patternSize);
  parameters->update(delta);
  energyFunction->clearPrecomputations();
  VecXt rxPlusDelta = getResidualValues(*energyFunction, patternSize);
  parameters->recoverState(state);
  delta = -1 * delta;
  parameters->update(delta);
  energyFunction->clearPrecomputations();
  VecXt rxMinusDelta = getResidualValues(*energyFunction, patternSize);

  VecXt diff = rxPlusDelta - rxMinusDelta;
  VecXt errs = diff - twoJDeltaX;
  double deltaNorm = std::sqrt(delta.getFrame().squaredNorm() +
                               delta.getPoint().squaredNorm());
  double err = errs.norm();
  double relErr = err / diff.norm();
  outputArray("errs.txt", errs.data(), errs.size());
  outputArray("diff.txt", diff.data(), diff.size());
  EXPECT_LE(relErr, ErrorBounds<T>::jacobianRelErr)
      << "err=" << err << " deltaNorm=" << deltaNorm
      << " diffNorm=" << diff.norm();
}

TEST_F(EnergyFunctionTest, arePredictionsCorrect) {
  const int PS = settings.residualPattern.pattern().size();
  int numResiduals = energyFunction->getResiduals().size();
  int numPoints = energyFunction->numPoints();

  DeltaParameterVector delta =
      sampleDelta(*frameParameterOrder, energyFunction->numPoints());
  const auto &residuals = energyFunction->getResiduals();
  MatXXt expectedJacobian = getExpectedJacobian();
  VecXt deltaVec(delta.getFrame().size() + delta.getPoint().size());
  deltaVec << delta.getFrame(), delta.getPoint();

  VecXt expectedPrediction = expectedJacobian * deltaVec;
  VecXt actualPrediction(PS * residuals.size());
  for (int ri = 0; ri < residuals.size(); ++ri) {
    VecRt expectedBlock = expectedPrediction.segment(ri * PS, PS);
    expectedBlock = expectedBlock.cwiseProduct(
        residuals[ri].getPixelDependentWeights().cwiseSqrt());
    expectedPrediction.segment(ri * PS, PS) = expectedBlock;
    actualPrediction.segment(ri * PS, PS) =
        energyFunction->getPredictedResidualIncrement(ri, delta);
  }

  double relPredictionErr = (expectedPrediction - actualPrediction).norm() /
                            expectedPrediction.norm();
  LOG(INFO) << "relative prediction err = " << relPredictionErr;
  EXPECT_LE(relPredictionErr, ErrorBounds<T>::predictionRelErr);
}

TEST_F(EnergyFunctionTest, isHessianCorrect) {
  static_assert(keyFramesCount >= 2);

  if (FLAGS_profile_only) {
    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    for (int i = 0; i < FLAGS_num_evaluations; ++i) {
      if (i % 20 == 0)
        std::cout << "i = " << i << std::endl;
      energyFunction->precomputeValuesAndDerivatives();
      Hessian actualHessian = energyFunction->getHessian();
      Gradient actualGradient = energyFunction->getGradient();
      energyFunction->clearPrecomputations();
    }
    end = std::chrono::system_clock::now();
    double totalTime = secondsBetween(start, end);
    LOG(INFO) << "time of hessian and gradient evaluation ("
              << FLAGS_num_evaluations << " times): " << totalTime;
    LOG(INFO) << totalTime / FLAGS_num_evaluations
              << " sec for one computation on average (" << FLAGS_opt_count
              << " optimized points, " << keyFramesCount << " keyframes)";
  } else {
    Hessian actualHessian = energyFunction->getHessian();

    const int PS = settings.residualPattern.pattern().size();
    int totalFrameParams = sndFrameDoF + (keyFramesCount - 2) * restFrameDoF;
    int numResiduals = energyFunction->getResiduals().size();
    int numPoints = energyFunction->numPoints();
    const auto &residuals = energyFunction->getResiduals();

    MatXXt expectedJacobian = getExpectedJacobian();
    VecXt weights(expectedJacobian.rows());
    ASSERT_EQ(weights.size(), PS * energyFunction->getResiduals().size());

    for (int i = 0; i < residuals.size(); ++i)
      weights.segment(i * PS, PS) =
          residuals[i].getHessianWeights(energyFunction->getResidualValues(i));

    LOG(INFO) << "jacobian size: " << expectedJacobian.rows() << " x "
              << expectedJacobian.cols();

    LOG(INFO) << "evaluating J^T J...";

    MatXX expectedHessian =
        (expectedJacobian.transpose() * weights.asDiagonal() * expectedJacobian)
            .cast<double>();

    MatXX expectedFrameFrame =
        expectedHessian.topLeftCorner(totalFrameParams, totalFrameParams);
    MatXX expectedFramePoint =
        expectedHessian.topRightCorner(totalFrameParams, numPoints);
    VecX expectedPointPoint =
        expectedHessian.bottomRightCorner(numPoints, numPoints).diagonal();

    MatXX actualFrameFrame = actualHessian.getFrameFrame().cast<double>();
    MatXX actualFramePoint = actualHessian.getFramePoint().cast<double>();
    VecX actualPointPoint = actualHessian.getPointPoint().cast<double>();

    LOG(INFO) << "NaNs in jacobian: " << countNaNs(expectedJacobian) << " of "
              << expectedJacobian.size();
    LOG(INFO) << "NaNs in expected FrameFrame: "
              << countNaNs(expectedFrameFrame) << " of "
              << expectedFrameFrame.size();
    LOG(INFO) << "NaNs in actual FrameFrame: " << countNaNs(actualFrameFrame)
              << " of " << actualFrameFrame.size();
    constexpr double relEps = 1e-8;
    double ffEps = (actualFrameFrame.norm() / actualFrameFrame.size()) * relEps;
    LOG(INFO) << "FrameFrame fill factor = "
              << fillFactor(actualFrameFrame, ffEps);
    double fpEps = (actualFramePoint.norm() / actualFramePoint.size()) * relEps;
    LOG(INFO) << "FramePoint fill factor = "
              << fillFactor(actualFramePoint, ffEps);

    ASSERT_EQ(expectedFrameFrame.rows(), actualFrameFrame.rows());
    ASSERT_EQ(expectedFrameFrame.cols(), actualFrameFrame.cols());
    double relFrameFrameErr = (expectedFrameFrame - actualFrameFrame).norm() /
                              expectedFrameFrame.norm();

    LOG(INFO) << "relative H_frameframe error = " << relFrameFrameErr << "\n";
    EXPECT_LE(relFrameFrameErr, ErrorBounds<T>::hessianRelErr);

    MatXX ffDiff = (actualFrameFrame - expectedFrameFrame);
    outputMatrix("ff_diff.txt", ffDiff);

    ASSERT_EQ(expectedFramePoint.rows(), actualFramePoint.rows());
    ASSERT_EQ(expectedFramePoint.cols(), actualFramePoint.cols());
    double relFramePointErr = (expectedFramePoint - actualFramePoint).norm() /
                              expectedFramePoint.norm();
    LOG(INFO) << "relative H_framepoint error = " << relFramePointErr << "\n";
    EXPECT_LE(relFramePointErr, ErrorBounds<T>::hessianRelErr);

    double relPointPointErr = (expectedPointPoint - actualPointPoint).norm() /
                              expectedPointPoint.norm();
    LOG(INFO) << "relative H_pointpoint error = " << relPointPointErr << "\n";
    EXPECT_LE(relPointPointErr, ErrorBounds<T>::hessianRelErr);
  }
}

TEST_F(EnergyFunctionTest, isGradientCorrect) {
  static_assert(keyFramesCount >= 2);
  Gradient actualGradient = energyFunction->getGradient();

  const int PS = settings.residualPattern.pattern().size();
  int numResiduals = energyFunction->getResiduals().size();
  int numPoints = energyFunction->numPoints();

  const auto &residuals = energyFunction->getResiduals();

  MatXXt expectedJacobian = getExpectedJacobian();
  VecXt weights(expectedJacobian.rows());
  VecXt values(expectedJacobian.rows());
  ASSERT_EQ(weights.size(), PS * energyFunction->getResiduals().size());
  for (int i = 0; i < residuals.size(); ++i) {
    VecRt curValues = energyFunction->getResidualValues(i);
    values.segment(i * PS, PS) = curValues;
    weights.segment(i * PS, PS) = residuals[i].getGradientWeights(curValues);
  }

  VecXt expectedGradient =
      expectedJacobian.transpose() * weights.asDiagonal() * values;
  VecXt expectedFrame =
      expectedGradient.head(frameParameterOrder->totalFrameParameters());
  VecXt expectedPoint = expectedGradient.tail(numPoints);

  double relFrameErr =
      (expectedFrame - actualGradient.getFrame()).norm() / expectedFrame.norm();
  double relPointErr =
      (expectedPoint - actualGradient.getPoint()).norm() / expectedPoint.norm();
  LOG(INFO) << "relative frame gradient err = " << relFrameErr;
  LOG(INFO) << "relative point gradient err = " << relPointErr;
  EXPECT_LE(relFrameErr, ErrorBounds<T>::gradientRelErr);
  EXPECT_LE(relPointErr, ErrorBounds<T>::gradientRelErr);
}

TEST_F(EnergyFunctionTest, doesOptimizationHelp) {
  constexpr int numIterations = 100;

  auto state = energyFunction->saveState();
  if (FLAGS_profile_only) {
    constexpr int shortNumIterations = 3;
    double totalTime = 0;
    for (int it = 0; it < FLAGS_num_evaluations; ++it) {
      if (it % 100 == 0)
        std::cout << "it = " << it << std::endl;
      TimePoint start, end;
      start = now();
      energyFunction->optimize(shortNumIterations);
      end = now();
      totalTime += secondsBetween(start, end);
      energyFunction->recoverState(state);
    }
    double avgTimeIter =
        totalTime / (FLAGS_num_evaluations * shortNumIterations);
    LOG(INFO) << "avg time per iteration = " << avgTimeIter;
    std::cout << "avg time per iteration = " << avgTimeIter << "\n";
  } else {
    double transErrBefore = transError();
    double rotErrBefore = rotError();

    energyFunction->optimize(numIterations);

    double transErrAfter = transError();
    double rotErrAfter = rotError();

    LOG(INFO) << "trans err before: " << transErrBefore;
    LOG(INFO) << "trans err after : " << transErrAfter;
    LOG(INFO) << "rot err before: " << rotErrBefore;
    LOG(INFO) << "rot err after : " << rotErrAfter;
    std::cout << "trans err before: " << transErrBefore << "\n";
    std::cout << "trans err after : " << transErrAfter << "\n";
    std::cout << "rot err before: " << rotErrBefore << "\n";
    std::cout << "rot err after : " << rotErrAfter << "\n";
  }
}

TEST_F(EnergyFunctionTest, doesCeresOptimizationHelp) {
  double transErrBefore = transError();
  double rotErrBefore = rotError();

  std::vector<KeyFrame *> kfPtrs;
  kfPtrs.reserve(keyFrames.size());
  for (auto &kf : keyFrames)
    kfPtrs.push_back(kf.get());

  BundleAdjusterCeres bundleAdjuster;
  bundleAdjuster.adjust(kfPtrs.data(), kfPtrs.size(),
                        settings.getBundleAdjusterSettings());

  double transErrAfter = transError();
  double rotErrAfter = rotError();

  LOG(INFO) << "Ceres optimization quality:";
  LOG(INFO) << "trans err before: " << transErrBefore;
  LOG(INFO) << "trans err after : " << transErrAfter;
  LOG(INFO) << "rot err before: " << rotErrBefore;
  LOG(INFO) << "rot err after : " << rotErrAfter;
  std::cout << "trans err before: " << transErrBefore << "\n";
  std::cout << "trans err after : " << transErrAfter << "\n";
  std::cout << "rot err before: " << rotErrBefore << "\n";
  std::cout << "rot err after : " << rotErrAfter << "\n";
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  return RUN_ALL_TESTS();
}
