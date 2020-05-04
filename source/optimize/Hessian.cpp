#include "optimize/Hessian.h"

namespace mdso::optimize {

Hessian::AccumulatedBlocks::AccumulatedBlocks(int numKeyFrames, int numCameras,
                                              int numPoints)
    : mMotionMotion(boost::extents[numKeyFrames - 1][numKeyFrames - 1])
    , mMotionAff(boost::extents[numKeyFrames - 1][numKeyFrames - 1][numCameras])
    , mAffAff(boost::extents[numKeyFrames - 1][numCameras][numKeyFrames - 1]
                            [numCameras])
    , mMotionPoint(boost::extents[numKeyFrames - 1][numPoints])
    , mAffPoint(boost::extents[numKeyFrames - 1][numCameras][numPoints])
    , mPointPoint(numPoints) {}

int Hessian::AccumulatedBlocks::numKeyFrames() const {
  return mAffPoint.shape()[0] + 1;
}

int Hessian::AccumulatedBlocks::numCameras() const {
  return mAffPoint.shape()[1];
}

int Hessian::AccumulatedBlocks::numPoints() const {
  return mAffPoint.shape()[2];
}

void Hessian::AccumulatedBlocks::add(
    const Residual::FrameFrameHessian &frameFrameHessian, int f1i, int f1ci,
    int f2i, int f2ci) {
  if (f1i < 1 || f2i < 1)
    return;

  if (f1i <= f2i) {
    motionMotion(f1i, f2i) += frameFrameHessian.qtqt;
    affAff(f1i, f1ci, f2i, f2ci) += frameFrameHessian.abab;
  } else {
    motionMotion(f2i, f1i) += frameFrameHessian.qtqt.transpose();
    affAff(f2i, f2ci, f1i, f1ci) += frameFrameHessian.abab.transpose();
  }

  motionAff(f1i, f2i, f2ci) += frameFrameHessian.qtab;
  if (f1i != f2i)
    motionAff(f2i, f1i, f1ci) += frameFrameHessian.abqt.transpose();
}

void Hessian::AccumulatedBlocks::add(
    const Residual::FramePointHessian &framePointHessian, int fi, int fci,
    int pi) {
  if (fi < 1)
    return;
  motionPoint(fi, pi) += framePointHessian.qtd;
  affPoint(fi, fci, pi) += framePointHessian.abd;
}

void Hessian::AccumulatedBlocks::add(
    const Residual &residual, const Residual::DeltaHessian &deltaHessian) {
  int hi = residual.hostInd(), hci = residual.hostCamInd(),
      ti = residual.targetInd(), tci = residual.targetCamInd(),
      pi = residual.pointInd();

  add(deltaHessian.hostHost, hi, hci, hi, hci);
  add(deltaHessian.hostTarget, hi, hci, ti, tci);
  add(deltaHessian.targetTarget, ti, tci, ti, tci);

  add(deltaHessian.hostPoint, hi, hci, pi);
  add(deltaHessian.targetPoint, ti, tci, pi);

  pointPoint(pi) += deltaHessian.pointPoint;
}
const Accumulator<Mat77t> &
Hessian::AccumulatedBlocks::motionMotion(int frameInd1, int frameInd2) const {
  CHECK_GE(frameInd1, 1);
  CHECK_LT(frameInd1, numKeyFrames());
  CHECK_GE(frameInd2, 1);
  CHECK_LT(frameInd2, numKeyFrames());
  return mMotionMotion[frameInd1 - 1][frameInd2 - 1];
}

Accumulator<Mat77t> &Hessian::AccumulatedBlocks::motionMotion(int frameInd1,
                                                              int frameInd2) {
  return const_cast<Accumulator<Mat77t> &>(
      const_cast<const Hessian::AccumulatedBlocks *>(this)->motionMotion(
          frameInd1, frameInd2));
}

const Accumulator<Mat72t> &
Hessian::AccumulatedBlocks::motionAff(int frameInd1, int frameInd2,
                                      int frameCamInd2) const {
  CHECK_GE(frameInd1, 1);
  CHECK_LT(frameInd1, numKeyFrames());
  CHECK_GE(frameInd2, 1);
  CHECK_LT(frameInd2, numKeyFrames());
  CHECK_GE(frameCamInd2, 0);
  CHECK_LT(frameCamInd2, numCameras());
  return mMotionAff[frameInd1 - 1][frameInd2 - 1][frameCamInd2];
}

Accumulator<Mat72t> &Hessian::AccumulatedBlocks::motionAff(int frameInd1,
                                                           int frameInd2,
                                                           int frameCamInd2) {
  return const_cast<Accumulator<Mat72t> &>(
      const_cast<const Hessian::AccumulatedBlocks *>(this)->motionAff(
          frameInd1, frameInd2, frameCamInd2));
}

const Accumulator<Mat22t> &
Hessian::AccumulatedBlocks::affAff(int frameInd1, int frameCamInd1,
                                   int frameInd2, int frameCamInd2) const {
  CHECK_GE(frameInd1, 1);
  CHECK_LT(frameInd1, numKeyFrames());
  CHECK_GE(frameCamInd1, 0);
  CHECK_LT(frameCamInd1, numCameras());
  CHECK_GE(frameInd2, 1);
  CHECK_LT(frameInd2, numKeyFrames());
  CHECK_GE(frameCamInd2, 0);
  CHECK_LT(frameCamInd2, numCameras());
  return mAffAff[frameInd1 - 1][frameCamInd1][frameInd2 - 1][frameCamInd2];
}

Accumulator<Mat22t> &Hessian::AccumulatedBlocks::affAff(int frameInd1,
                                                        int frameCamInd1,
                                                        int frameInd2,
                                                        int frameCamInd2) {
  return const_cast<Accumulator<Mat22t> &>(
      const_cast<const Hessian::AccumulatedBlocks *>(this)->affAff(
          frameInd1, frameCamInd1, frameInd2, frameCamInd2));
}

const Accumulator<Vec7t> &
Hessian::AccumulatedBlocks::motionPoint(int frameInd, int pointInd) const {
  CHECK_GE(frameInd, 1);
  CHECK_LT(frameInd, numKeyFrames());
  CHECK_GE(pointInd, 0);
  CHECK_LT(pointInd, numPoints());
  return mMotionPoint[frameInd - 1][pointInd];
}

Accumulator<Vec7t> &Hessian::AccumulatedBlocks::motionPoint(int frameInd,
                                                            int pointInd) {
  return const_cast<Accumulator<Vec7t> &>(
      const_cast<const Hessian::AccumulatedBlocks *>(this)->motionPoint(
          frameInd, pointInd));
}

const Accumulator<Vec2t> &
Hessian::AccumulatedBlocks::affPoint(int frameInd, int frameCamInd,
                                     int pointInd) const {
  CHECK_GE(frameInd, 1);
  CHECK_LT(frameInd, numKeyFrames());
  CHECK_GE(frameCamInd, 0);
  CHECK_LT(frameCamInd, numCameras());
  CHECK_GE(pointInd, 0);
  CHECK_LT(pointInd, numPoints());
  return mAffPoint[frameInd - 1][frameCamInd][pointInd];
}
Accumulator<Vec2t> &Hessian::AccumulatedBlocks::affPoint(int frameInd,
                                                         int frameCamInd,
                                                         int pointInd) {
  return const_cast<Accumulator<Vec2t> &>(
      const_cast<const Hessian::AccumulatedBlocks *>(this)->affPoint(
          frameInd, frameCamInd, pointInd));
}

const Accumulator<T> &
Hessian::AccumulatedBlocks::pointPoint(int pointInd) const {
  CHECK_GE(pointInd, 0);
  CHECK_LT(pointInd, numPoints());
  return mPointPoint[pointInd];
}

Accumulator<T> &Hessian::AccumulatedBlocks::pointPoint(int pointInd) {
  return const_cast<Accumulator<T> &>(
      const_cast<const Hessian::AccumulatedBlocks *>(this)->pointPoint(
          pointInd));
}

Hessian::Hessian(const AccumulatedBlocks &accumulatedBlocks,
                 const Parameters::Jacobians &parameterJacobians,
                 const Settings::Optimization &settings)
    : frameParameterOrder(accumulatedBlocks.numKeyFrames(),
                          accumulatedBlocks.numCameras())
    , frameFrame(frameParameterOrder.totalFrameParameters(),
                 frameParameterOrder.totalFrameParameters())
    , framePoint(frameParameterOrder.totalFrameParameters(),
                 accumulatedBlocks.numPoints())
    , pointPoint(accumulatedBlocks.numPoints())
    , settings(settings) {
  frameFrame.setZero();
  framePoint.setZero();
  pointPoint.setZero();

  if (accumulatedBlocks.motionMotion(1, 1).wasUsed())
    sndSndBlock() = parameterJacobians.dSecondFrame().transpose() *
                    accumulatedBlocks.motionMotion(1, 1).accumulated() *
                    parameterJacobians.dSecondFrame();

  for (int fi = 2; fi < accumulatedBlocks.numKeyFrames(); ++fi)
    if (accumulatedBlocks.motionMotion(1, fi).wasUsed())
      sndRestBlock(fi) = parameterJacobians.dSecondFrame().transpose() *
                         accumulatedBlocks.motionMotion(1, fi).accumulated() *
                         parameterJacobians.dOtherFrame(fi);

  for (int fi = 1; fi < accumulatedBlocks.numKeyFrames(); ++fi)
    for (int ci = 0; ci < accumulatedBlocks.numCameras(); ++ci)
      if (accumulatedBlocks.motionAff(1, fi, ci).wasUsed())
        sndAffBlock(fi, ci) =
            parameterJacobians.dSecondFrame().transpose() *
            accumulatedBlocks.motionAff(1, fi, ci).accumulated();

  for (int fi1 = 2; fi1 < accumulatedBlocks.numKeyFrames(); ++fi1)
    for (int fi2 = 2; fi2 < accumulatedBlocks.numKeyFrames(); ++fi2)
      if (accumulatedBlocks.motionMotion(fi1, fi2).wasUsed())
        restRestBlock(fi1, fi2) =
            parameterJacobians.dOtherFrame(fi1).transpose() *
            accumulatedBlocks.motionMotion(fi1, fi2).accumulated() *
            parameterJacobians.dOtherFrame(fi2);

  for (int fi1 = 2; fi1 < accumulatedBlocks.numKeyFrames(); ++fi1)
    for (int fi2 = 1; fi2 < accumulatedBlocks.numKeyFrames(); ++fi2)
      for (int ci = 0; ci < accumulatedBlocks.numCameras(); ++ci)
        if (accumulatedBlocks.motionAff(fi1, fi2, ci).wasUsed())
          restAffBlock(fi1, fi2, ci) =
              parameterJacobians.dOtherFrame(fi1).transpose() *
              accumulatedBlocks.motionAff(fi1, fi2, ci).accumulated();

  for (int fi1 = 1; fi1 < accumulatedBlocks.numKeyFrames(); ++fi1)
    for (int ci1 = 0; ci1 < accumulatedBlocks.numCameras(); ++ci1)
      for (int fi2 = 1; fi2 < accumulatedBlocks.numKeyFrames(); ++fi2)
        for (int ci2 = 0; ci2 < accumulatedBlocks.numCameras(); ++ci2)
          if (accumulatedBlocks.affAff(fi1, ci1, fi2, ci2).wasUsed())
            affAffBlock(fi1, ci1, fi2, ci2) =
                accumulatedBlocks.affAff(fi1, ci1, fi2, ci2).accumulated();

  for (int pi = 0; pi < accumulatedBlocks.numPoints(); ++pi)
    if (accumulatedBlocks.motionPoint(1, pi).wasUsed())
      sndPointBlock(pi) = parameterJacobians.dSecondFrame().transpose() *
                          accumulatedBlocks.motionPoint(1, pi).accumulated();

  for (int fi = 2; fi < accumulatedBlocks.numKeyFrames(); ++fi)
    for (int pi = 0; pi < accumulatedBlocks.numPoints(); ++pi)
      if (accumulatedBlocks.motionPoint(fi, pi).wasUsed())
        restPointBlock(fi, pi) =
            parameterJacobians.dOtherFrame(fi).transpose() *
            accumulatedBlocks.motionPoint(fi, pi).accumulated();

  for (int fi = 1; fi < accumulatedBlocks.numKeyFrames(); ++fi)
    for (int ci = 0; ci < accumulatedBlocks.numCameras(); ++ci)
      for (int pi = 0; pi < accumulatedBlocks.numPoints(); ++pi)
        if (accumulatedBlocks.affPoint(fi, ci, pi).wasUsed())
          affPointBlock(fi, ci, pi) =
              accumulatedBlocks.affPoint(fi, ci, pi).accumulated();

  for (int pi = 0; pi < accumulatedBlocks.numPoints(); ++pi)
    if (accumulatedBlocks.pointPoint(pi).wasUsed())
      pointPointBlock(pi) = accumulatedBlocks.pointPoint(pi).accumulated();

  fillLowerBlocks();
}

void Hessian::fillLowerBlocks() {
  int sz = frameFrame.rows();
  for (int i = 0; i < sz; ++i)
    frameFrame.block(i + 1, i, sz - i - 1, 1) =
        frameFrame.block(i, i + 1, 1, sz - i - 1).transpose();
}

Hessian Hessian::levenbergMarquardtDamp(double lambda) const {
  Hessian result = *this;
  result.frameFrame.diagonal() *= 1 + lambda;
  result.pointPoint *= 1 + lambda;
  return result;
}

std::vector<int> indsBigger(const VecXt &vec, T threshold) {
  std::vector<int> inds;
  inds.reserve(vec.size());
  for (int i = 0; i < vec.size(); ++i)
    if (vec[i] > threshold)
      inds.push_back(i);

  return inds;
}

// slicing is introduced in Eigen 3.4 only, not the current stable release
VecXt sliceInds(const VecXt &vec, const std::vector<int> &inds) {
  VecXt sliced(inds.size());
  for (int i = 0; i < inds.size(); ++i)
    sliced[i] = vec[inds[i]];
  return sliced;
}

MatXXt sliceCols(const MatXXt &mat, const std::vector<int> &inds) {
  MatXXt sliced(mat.rows(), inds.size());
  for (int i = 0; i < inds.size(); ++i)
    sliced.col(i) = mat.col(inds[i]);
  return sliced;
}

void extendWithZeros(VecXt &vec, const std::vector<int> &indsPresent,
                     int newSize) {
  VecXt extended(newSize);
  extended.setZero();
  for (int i = 0; i < indsPresent.size(); ++i)
    extended[indsPresent[i]] = vec[i];
  vec = extended;
}

std::vector<int> subtractSet(const std::vector<int> &inds1,
                             const std::vector<int> &inds2) {
  std::vector<int> result(inds1.size());
  auto newEnd = std::set_difference(inds1.begin(), inds1.end(), inds2.begin(),
                                    inds2.end(), result.begin());
  result.resize(newEnd - result.begin());
  return result;
}

DeltaParameterVector Hessian::solve(const Gradient &gradient,
                                    const int *excludedPointInds,
                                    int excludedPointIndsSize) const {
  int totalPoints = pointPoint.size();
  std::vector<int> pointIndsBigger =
      indsBigger(pointPoint, settings.pointPointThres);
  std::vector<int> pointIndsUsed;
  if (excludedPointInds) {
    CHECK_GE(excludedPointIndsSize, 0);
    pointIndsUsed =
        subtractSet(pointIndsBigger,
                    std::vector(excludedPointInds,
                                excludedPointInds + excludedPointIndsSize));
  } else {
    pointIndsUsed = pointIndsBigger;
  }
  VecXt pointPointSliced = sliceInds(pointPoint, pointIndsUsed);
  MatXXt framePointSliced = sliceCols(framePoint, pointIndsUsed);
  VecXt gradPointSliced = sliceInds(gradient.getPoint(), pointIndsUsed);

  VecXt pointPointInv = pointPointSliced.cwiseInverse();
  CHECK_EQ((!pointPointInv.array().isFinite()).count(), 0);
  MatXXt hessianSchur = frameFrame - framePointSliced *
                                         pointPointInv.asDiagonal() *
                                         framePointSliced.transpose();
  VecXt pointPointInv_point = pointPointInv.cwiseProduct(-gradPointSliced);
  VecXt gradientSchur =
      -gradient.getFrame() - framePointSliced * pointPointInv_point;
  VecXt deltaFrame = hessianSchur.ldlt().solve(gradientSchur);
  VecXt deltaPoint =
      pointPointInv_point -
      pointPointInv.cwiseProduct(framePointSliced.transpose() * deltaFrame);

  T frameErr = (frameFrame * deltaFrame + framePointSliced * deltaPoint +
                gradient.getFrame())
                   .norm() /
               gradient.getFrame().norm();
  T pointErr = (framePointSliced.transpose() * deltaFrame +
                pointPointSliced.cwiseProduct(deltaPoint) + gradPointSliced)
                   .norm() /
               gradPointSliced.norm();
  LOG(INFO) << "rel solution frame err = " << frameErr
            << ", rel point err = " << pointErr;

  extendWithZeros(deltaPoint, pointIndsUsed, totalPoints);

  return DeltaParameterVector(frameParameterOrder.numKeyFrames(),
                              frameParameterOrder.numCameras(), deltaFrame,
                              deltaPoint);
}

T Hessian::applyQuadraticForm(const DeltaParameterVector &delta) const {
  const VecXt &deltaFrame = delta.getFrame();
  const VecXt &deltaPoint = delta.getPoint();
  return (deltaFrame.transpose() * frameFrame * deltaFrame +
          2 * deltaFrame.transpose() * framePoint * deltaPoint)
             .value() +
         deltaPoint.dot(pointPoint.cwiseProduct(deltaPoint));
}

} // namespace mdso::optimize
