#include "optimize/Hessian.h"

namespace mdso::optimize {

Hessian::AccumulatedBlocks::AccumulatedBlocks(int numKeyFrames, int numCameras,
                                              int numPoints)
    : motionMotion(boost::extents[numKeyFrames - 1][numKeyFrames - 1])
    , motionAff(boost::extents[numKeyFrames - 1][numKeyFrames - 1][numCameras])
    , affAff(boost::extents[numKeyFrames - 1][numCameras][numKeyFrames - 1]
                           [numCameras])
    , motionPoint(boost::extents[numKeyFrames - 1][numPoints])
    , affPoint(boost::extents[numKeyFrames - 1][numCameras][numPoints])
    , pointPoint(numPoints) {}

int Hessian::AccumulatedBlocks::numKeyFrames() const {
  return affPoint.shape()[0] + 1;
}

int Hessian::AccumulatedBlocks::numCameras() const {
  return affPoint.shape()[1];
}

int Hessian::AccumulatedBlocks::numPoints() const {
  return affPoint.shape()[2];
}

void Hessian::AccumulatedBlocks::add(
    const Residual::FrameFrameHessian &frameFrameHessian, int f1i, int f1ci,
    int f2i, int f2ci) {
  if (f1i < 0 || f2i < 0)
    return;

  if (f1i <= f2i) {
    motionMotion[f1i][f2i] += frameFrameHessian.qtqt;
    affAff[f1i][f1ci][f2i][f2ci] += frameFrameHessian.abab;
  } else {
    motionMotion[f2i][f1i] += frameFrameHessian.qtqt.transpose();
    affAff[f2i][f2ci][f1i][f1ci] += frameFrameHessian.abab.transpose();
  }

  motionAff[f1i][f2i][f2ci] += frameFrameHessian.qtab;
  if (f1i != f2i)
    motionAff[f2i][f1i][f1ci] += frameFrameHessian.abqt.transpose();
}

void Hessian::AccumulatedBlocks::add(
    const Residual::FramePointHessian &framePointHessian, int fi, int fci,
    int pi) {
  if (fi < 0)
    return;
  motionPoint[fi][pi] += framePointHessian.qtd;
  affPoint[fi][fci][pi] += framePointHessian.abd;
}

void Hessian::AccumulatedBlocks::add(
    const Residual &residual, const Residual::DeltaHessian &deltaHessian) {
  int hi = residual.hostInd() - 1, hci = residual.hostCamInd(),
      ti = residual.targetInd() - 1, tci = residual.targetCamInd(),
      pi = residual.pointInd();

  add(deltaHessian.hostHost, hi, hci, hi, hci);
  add(deltaHessian.hostTarget, hi, hci, ti, tci);
  add(deltaHessian.targetTarget, ti, tci, ti, tci);

  add(deltaHessian.hostPoint, hi, hci, pi);
  add(deltaHessian.targetPoint, ti, tci, pi);

  pointPoint[pi] += deltaHessian.pointPoint;
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

  if (accumulatedBlocks.motionMotion[0][0].wasUsed())
    sndSndBlock() = parameterJacobians.dSecondFrame.transpose() *
                    accumulatedBlocks.motionMotion[0][0].accumulated() *
                    parameterJacobians.dSecondFrame;

  for (int fi = 2; fi < accumulatedBlocks.numKeyFrames(); ++fi)
    if (accumulatedBlocks.motionMotion[0][fi - 1].wasUsed())
      sndRestBlock(fi) =
          parameterJacobians.dSecondFrame.transpose() *
          accumulatedBlocks.motionMotion[0][fi - 1].accumulated() *
          parameterJacobians.dRestFrames[fi - 2];

  for (int fi = 1; fi < accumulatedBlocks.numKeyFrames(); ++fi)
    for (int ci = 0; ci < accumulatedBlocks.numCameras(); ++ci)
      if (accumulatedBlocks.motionAff[0][fi - 1][ci].wasUsed())
        sndAffBlock(fi, ci) =
            parameterJacobians.dSecondFrame.transpose() *
            accumulatedBlocks.motionAff[0][fi - 1][ci].accumulated();

  for (int fi1 = 2; fi1 < accumulatedBlocks.numKeyFrames(); ++fi1)
    for (int fi2 = 2; fi2 < accumulatedBlocks.numKeyFrames(); ++fi2)
      if (accumulatedBlocks.motionMotion[fi1 - 1][fi2 - 1].wasUsed())
        restRestBlock(fi1, fi2) =
            parameterJacobians.dRestFrames[fi1 - 2].transpose() *
            accumulatedBlocks.motionMotion[fi1 - 1][fi2 - 1].accumulated() *
            parameterJacobians.dRestFrames[fi2 - 2];

  for (int fi1 = 2; fi1 < accumulatedBlocks.numKeyFrames(); ++fi1)
    for (int fi2 = 1; fi2 < accumulatedBlocks.numKeyFrames(); ++fi2)
      for (int ci = 0; ci < accumulatedBlocks.numCameras(); ++ci)
        if (accumulatedBlocks.motionAff[fi1 - 1][fi2 - 1][ci].wasUsed())
          restAffBlock(fi1, fi2, ci) =
              parameterJacobians.dRestFrames[fi1 - 2].transpose() *
              accumulatedBlocks.motionAff[fi1 - 1][fi2 - 1][ci].accumulated();

  for (int fi1 = 1; fi1 < accumulatedBlocks.numKeyFrames(); ++fi1)
    for (int ci1 = 0; ci1 < accumulatedBlocks.numCameras(); ++ci1)
      for (int fi2 = 1; fi2 < accumulatedBlocks.numKeyFrames(); ++fi2)
        for (int ci2 = 0; ci2 < accumulatedBlocks.numCameras(); ++ci2)
          if (accumulatedBlocks.affAff[fi1 - 1][ci1][fi2 - 1][ci2].wasUsed())
            affAffBlock(fi1, ci1, fi2, ci2) =
                accumulatedBlocks.affAff[fi1 - 1][ci1][fi2 - 1][ci2]
                    .accumulated();

  for (int pi = 0; pi < accumulatedBlocks.numPoints(); ++pi)
    if (accumulatedBlocks.motionPoint[0][pi].wasUsed())
      sndPointBlock(pi) = parameterJacobians.dSecondFrame.transpose() *
                          accumulatedBlocks.motionPoint[0][pi].accumulated();

  for (int fi = 2; fi < accumulatedBlocks.numKeyFrames(); ++fi)
    for (int pi = 0; pi < accumulatedBlocks.numPoints(); ++pi)
      if (accumulatedBlocks.motionPoint[fi - 1][pi].wasUsed())
        restPointBlock(fi, pi) =
            parameterJacobians.dRestFrames[fi - 2].transpose() *
            accumulatedBlocks.motionPoint[fi - 1][pi].accumulated();

  for (int fi = 1; fi < accumulatedBlocks.numKeyFrames(); ++fi)
    for (int ci = 0; ci < accumulatedBlocks.numCameras(); ++ci)
      for (int pi = 0; pi < accumulatedBlocks.numPoints(); ++pi)
        if (accumulatedBlocks.affPoint[fi - 1][ci][pi].wasUsed())
          affPointBlock(fi, ci, pi) =
              accumulatedBlocks.affPoint[fi - 1][ci][pi].accumulated();

  for (int pi = 0; pi < accumulatedBlocks.numPoints(); ++pi)
    if (accumulatedBlocks.pointPoint[pi].wasUsed())
      pointPointBlock(pi) = accumulatedBlocks.pointPoint[pi].accumulated();

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
  result.frameFrame.diagonal() *= (1 + lambda);
  result.pointPoint *= (1 + lambda);
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

DeltaParameterVector Hessian::solve(const Gradient &gradient) const {
  int totalPoints = pointPoint.size();
  std::vector<int> pointIndsUsed =
      indsBigger(pointPoint, settings.pointPointThres);
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

} // namespace mdso::optimize
