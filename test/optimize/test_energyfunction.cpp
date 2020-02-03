#include "MultiFovReader.h"
#include "optimize/EnergyFunction.h"
#include "system/IdentityPreprocessor.h"
#include "util/flags.h"
#include <Eigen/Eigen>
#include <Eigen/Sparse>
#include <gtest/gtest.h>

DEFINE_string(mfov_dir, "/shared/datasets/mfov",
              "Root folder of the MultiFoV dataset.");

DEFINE_int32(opt_count, 100, "Number of optimized points in the problem.");
DEFINE_bool(get_fill_factor, false,
            "If set to true, HessianTest only returns H_framepoint fill factor "
            "without hessian correctness checking.");

using namespace mdso;
using namespace mdso::optimize;

using Triplet = Eigen::Triplet<T>;
using MatSp = Eigen::SparseMatrix<T>;

template <typename T> struct ErrorBounds;

template <> struct ErrorBounds<float> {
  static constexpr double hessianRelErr = 1;
};

template <> struct ErrorBounds<double> {
  static constexpr double hessianRelErr = 1e-7;
};

template <typename MatrixT>
void setBlock(std::vector<Triplet> &triplets, const MatrixT &block,
              int startRow, int startCol) {
  for (int r = startRow, br = 0; br < block.rows(); ++r, ++br)
    for (int c = startCol, bc = 0; bc < block.cols(); ++c, ++bc)
      triplets.emplace_back(r, c, block(br, bc));
}

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

  EnergyFunctionTest()
      : pointsPerFrame(FLAGS_opt_count / keyFramesCount) {}

  void SetUp() override {
    settings = getFlaggedSettings();
    settings.setMaxOptimizedPoints(pointsPerFrame * keyFramesCount);
    settings.keyFrame.setImmaturePointsNum(pointsPerFrame);
    residualSettings = settings.getResidualSettings();

    reader.reset(new MultiFovReader(FLAGS_mfov_dir));
    SE3 bodyToFrame = SE3::sampleUniform(mt);
    cam.reset(new CameraBundle(&bodyToFrame, reader->cam.get(), 1));
    PixelSelector pixelSelector;
    pixelSelector.initialize(reader->getFrame(keyFrameNums[0]),
                             settings.keyFrame.immaturePointsNum());
    keyFrames.reserve(keyFramesCount);
    for (int i = 0; i < keyFramesCount; ++i) {
      Timestamp ts = keyFrameNums[i];
      AffLight lightWorldToF = sampleAffLight<double>(settings.affineLight, mt);
      cv::Mat3b coloredFrame = reader->getFrame(keyFrameNums[i]);
      coloredFrame = lightWorldToF(coloredFrame);
      std::unique_ptr<PreKeyFrame> pkf(
          new PreKeyFrame(nullptr, cam.get(), &idPrep, &coloredFrame, i, &ts,
                          settings.pyramid));
      keyFrames.emplace_back(new KeyFrame(std::move(pkf), &pixelSelector,
                                          settings.keyFrame,
                                          settings.getPointTracerSettings()));
      keyFrames.back()->frames[0].lightWorldToThis = lightWorldToF;
      keyFrames.back()->thisToWorld.setValue(
          reader->getWorldToFrameGT(keyFrameNums[i]).inverse());
      cv::Mat1d depths = reader->getDepths(keyFrameNums[i]);
      for (ImmaturePoint &ip : keyFrames.back()->frames[0].immaturePoints) {
        cv::Point p = toCvPoint(ip.p);
        checkBounds(depths, p);
        ip.setTrueDepth(depths(p), settings.pointTracer);
        keyFrames.back()->frames[0].optimizedPoints.emplace_back(ip);
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
    energyFunction.reset(new EnergyFunction(cam.get(), kfPtrs.data(),
                                            kfPtrs.size(), residualSettings));
  }

  void setFrame(MatXXt &jacobian, int residualInd, int frameInd,
                const MatR4t &dr_dq, const MatR3t &dr_dt,
                const MatR2t &dr_daff) {
    if (frameInd == 0)
      return;
    if (frameInd == 1)
      setSecondFrame(jacobian, residualInd, dr_dq, dr_dt, dr_daff);
    else
      setOtherFrame(jacobian, residualInd, frameInd, dr_dq, dr_dt, dr_daff);
  }

  std::unique_ptr<MultiFovReader> reader;
  std::unique_ptr<CameraBundle> cam;
  std::vector<std::unique_ptr<KeyFrame>> keyFrames;
  std::unique_ptr<EnergyFunction> energyFunction;
  std::unique_ptr<SO3xS2Parametrization> secondFrameParam;
  std::vector<RightExpParametrization<SE3t>> restFrameParams;
  Settings settings;
  ResidualSettings residualSettings;
  IdentityPreprocessor idPrep;
  std::mt19937 mt;
  int pointsPerFrame;

private:
  void setSecondFrame(MatXXt &jacobian, int residualInd, const MatR4t &dr_dq,
                      const MatR3t &dr_dt, const MatR2t &dr_daff) {
    int PS = settings.residualPattern.pattern().size();
    int startRow = residualInd * PS;
    MatR7t dr_dqt(PS, 7);
    dr_dqt << dr_dq, dr_dt;
    Mat75t diffPlus = secondFrameParam->diffPlus();
    MatR5t dr_dqt_tang = dr_dqt * diffPlus;
    jacobian.block(startRow, 0, PS, sndDoF) = dr_dqt_tang;
    jacobian.block(startRow, sndDoF, PS, 2) = dr_daff;
  }

  void setOtherFrame(MatXXt &jacobian, int residualInd, int frameInd,
                     const MatR4t &dr_dq, const MatR3t &dr_dt,
                     const MatR2t &dr_daff) {
    CHECK_GE(frameInd, 2);
    int PS = settings.residualPattern.pattern().size();
    int startRow = residualInd * PS,
        startCol = sndFrameDoF + restFrameDoF * (frameInd - 2);
    MatR7t dr_dqt(PS, 7);
    dr_dqt << dr_dq, dr_dt;
    Mat76t diffPlus = restFrameParams[frameInd - 2].diffPlus();
    MatR6t dr_dqt_tang = dr_dqt * diffPlus;

    jacobian.block(startRow, startCol, PS, restDoF) = dr_dqt_tang;
    jacobian.block(startRow, startCol + restDoF, PS, 2) = dr_daff;
  }
};

template <typename MatrixT>
MatrixT weightRows(const MatrixT &mat, const VecRt &weights) {
  return weights.asDiagonal() * mat;
}

double fillFactor(const MatXXt &mat, double eps) {
  return double((mat.array().abs() >= eps).count()) / mat.size();
}

template <typename MatrixT> int countNaNs(const MatrixT &mat) {
  return (mat.array() != mat.array()).count();
}

TEST_F(EnergyFunctionTest, isHessianCorrect) {
  static_assert(keyFramesCount >= 2);

  if (FLAGS_get_fill_factor) {
    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    EnergyFunction::Hessian hessian = energyFunction->getHessian();
    end = std::chrono::system_clock::now();
    LOG(INFO) << "hessian evaluation time = "
              << std::chrono::duration_cast<std::chrono::nanoseconds>(end -
                                                                      start)
                         .count() *
                     1e-9
              << " seconds\n";

    {
      std::chrono::time_point<std::chrono::system_clock> start, end;
      start = std::chrono::system_clock::now();
      MatXXt schurCompl = hessian.framePoint * hessian.pointPoint.asDiagonal() *
                          hessian.framePoint.transpose();
      end = std::chrono::system_clock::now();
      LOG(INFO) << "Schur complement evaluation time = "
                << std::chrono::duration_cast<std::chrono::nanoseconds>(end -
                                                                        start)
                           .count() *
                       1e-9
                << " seconds\n";
    }

    constexpr double eps = 1e-5;
    LOG(INFO) << "H_frameframe fill factor = "
              << fillFactor(hessian.frameFrame, eps) << "\n";
    LOG(INFO) << "H_framepoint fill factor = "
              << fillFactor(hessian.framePoint, eps) << "\n";
    return;
  }

  const int PS = residualSettings.residualPattern.pattern().size();
  int totalFrameParams = sndFrameDoF + (keyFramesCount - 2) * restFrameDoF;
  MatXXt expectedJacobian(PS * energyFunction->getResiduals().size(),
                          totalFrameParams +
                              energyFunction->getOptimizedPoints().size());
  std::cout << "evaluating jacobian: 0% ... ";
  std::cout.flush();
  int percent = 0, step = 10;
  int numResiduals = energyFunction->getResiduals().size();
  for (int ri = 0; ri < numResiduals; ++ri) {
    const Residual &res = energyFunction->getResiduals()[ri];
    int hi = res.hostInd(), ti = res.targetInd();
    int pi = res.pointInd();

    KeyFrame *host = keyFrames[hi].get();
    KeyFrame *target = keyFrames[ti].get();
    SE3t hostToWorld = host->thisToWorld().cast<T>();
    SE3t targetToWorld = target->thisToWorld().cast<T>();
    SE3t hostFrameToBody = cam->bundle[0].thisToBody.cast<T>();
    SE3t targetBodyToFrame = cam->bundle[0].bodyToThis.cast<T>();

    AffLightT lightHostToTarget = (target->frames[0].lightWorldToThis *
                                   host->frames[0].lightWorldToThis.inverse())
                                      .cast<T>();

    MotionDerivatives dHostToTarget(hostFrameToBody, hostToWorld, targetToWorld,
                                    targetBodyToFrame);
    SE3t hostToTarget = targetBodyToFrame * targetToWorld.inverse() *
                        hostToWorld * hostFrameToBody;

    VecRt values = res.getValues(hostToTarget, lightHostToTarget);
    VecRt weights = res.getHessianWeights(values);
    VecRt sqrtWeights = weights.array().sqrt().matrix();

    Residual::Jacobian rj = res.getJacobian(
        hostToTarget, dHostToTarget, host->frames[0].lightWorldToThis.cast<T>(),
        lightHostToTarget);
    Residual::DeltaHessian dh = res.getDeltaHessian(values, rj);

    setFrame(expectedJacobian, ri, hi,
             weightRows(rj.dr_dq_host(PS), sqrtWeights),
             weightRows(rj.dr_dt_host(PS), sqrtWeights),
             weightRows(rj.dr_daff_host(PS), sqrtWeights));
    setFrame(expectedJacobian, ri, ti,
             weightRows(rj.dr_dq_target(PS), sqrtWeights),
             weightRows(rj.dr_dt_target(PS), sqrtWeights),
             weightRows(rj.dr_daff_target(PS), sqrtWeights));
    int depthCol = totalFrameParams + pi;
    ASSERT_LT(depthCol, expectedJacobian.cols());
    VecRt JD = rj.dr_dlogd(PS);
    auto weightedJD = weightRows(JD, sqrtWeights);

    expectedJacobian.block(ri * PS, depthCol, PS, 1) = weightedJD;

    int oldPercent = percent;
    percent = double(ri + 1) / numResiduals * 100.0;
    if (percent == 100)
      std::cout << "100%" << std::endl;
    else if (oldPercent != percent && percent % step == 0) {
      std::cout << percent << "% ... ";
      std::cout.flush();
    }
  }

  int numPoints = energyFunction->getOptimizedPoints().size();

  LOG(INFO) << "jacobian size: " << expectedJacobian.rows() << " x "
            << expectedJacobian.cols();

  LOG(INFO) << "evaluating J^T J...";

  MatXX expectedHessian =
      (expectedJacobian.transpose() * expectedJacobian).cast<double>();

  MatXX expectedFrameFrame =
      expectedHessian.topLeftCorner(totalFrameParams, totalFrameParams);
  MatXX expectedFramePoint =
      expectedHessian.topRightCorner(totalFrameParams, numPoints);
  VecX expectedPointPoint =
      expectedHessian.bottomRightCorner(numPoints, numPoints).diagonal();

  EnergyFunction::Hessian actualHessian = energyFunction->getHessian();

  MatXX actualFrameFrame = actualHessian.frameFrame.cast<double>();
  MatXX actualFramePoint = actualHessian.framePoint.cast<double>();
  VecX actualPointPoint = actualHessian.pointPoint.cast<double>();

  LOG(INFO) << "NaNs in jacobian: " << countNaNs(expectedJacobian) << " of "
            << expectedJacobian.size();
  LOG(INFO) << "NaNs in expected FrameFrame: " << countNaNs(expectedFrameFrame)
            << " of " << expectedFrameFrame.size();
  LOG(INFO) << "NaNs in actual FrameFrame: " << countNaNs(actualFrameFrame)
            << " of " << actualFrameFrame.size();

  ASSERT_EQ(expectedFrameFrame.rows(), actualFrameFrame.rows());
  ASSERT_EQ(expectedFrameFrame.cols(), actualFrameFrame.cols());
  double relFrameFrameErr = (expectedFrameFrame - actualFrameFrame).norm() /
                            expectedFrameFrame.norm();

  LOG(INFO) << "relative H_frameframe error = " << relFrameFrameErr << "\n";
  EXPECT_LE(relFrameFrameErr, ErrorBounds<T>::hessianRelErr);

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

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  return RUN_ALL_TESTS();
}
