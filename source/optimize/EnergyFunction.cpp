#include "optimize/EnergyFunction.h"
#include "optimize/Accumulator.h"

#define PH (settings.residualPattern.height)

namespace mdso::optimize {

EnergyFunction::OptimizationParams::OptimizationParams(
    CameraBundle *cam, const std::vector<KeyFrame *> &keyFrames)
    : secondFrame(keyFrames[0]->thisToWorld(), keyFrames[1]->thisToWorld())
    , lightWorldToFrame(
          boost::extents[keyFrames.size() - 1][cam->bundle.size()]) {
  restFrames.reserve(keyFrames.size() - 2);
  for (int i = 2; i < keyFrames.size(); ++i)
    restFrames.emplace_back(keyFrames[i]->thisToWorld().cast<T>());

  for (int fi = 1; fi < keyFrames.size(); ++fi)
    for (int ci = 0; ci < cam->bundle.size(); ++ci)
      lightWorldToFrame[fi - 1][ci] =
          keyFrames[fi]->frames[ci].lightWorldToThis.cast<T>();
}

EnergyFunction::EnergyFunction(CameraBundle *camBundle, KeyFrame **newKeyFrames,
                               int numKeyFrames,
                               const ResidualSettings &settings)
    : lossFunction(new ceres::HuberLoss(settings.intensity.outlierDiff))
    , cam(camBundle)
    , keyFrames(newKeyFrames, newKeyFrames + numKeyFrames)
    , optimizationParams(camBundle, keyFrames)
    , hostToTarget(boost::extents[numKeyFrames][cam->bundle.size()]
                                 [numKeyFrames][cam->bundle.size()])
    , hostToTargetDiff(boost::extents[numKeyFrames][cam->bundle.size()]
                                     [numKeyFrames][cam->bundle.size()])
    , lightHostToTarget(boost::extents[numKeyFrames][cam->bundle.size()]
                                      [numKeyFrames][cam->bundle.size()])
    , settings(settings) {
  CHECK(numKeyFrames >= 2);

  resetPrecomputations();

  int pointInd = 0;
  for (int hostInd = 0; hostInd < numKeyFrames; ++hostInd)
    for (int hostCamInd = 0; hostCamInd < cam->bundle.size(); ++hostCamInd) {
      for (OptimizedPoint &op :
           keyFrames[hostInd]->frames[hostCamInd].optimizedPoints) {
        if (op.state != OptimizedPoint::ACTIVE)
          continue;
        Vec3t ray = (op.depth() * op.dir).cast<T>();
        bool hasResiduals = false;
        for (int targetInd = 0; targetInd < numKeyFrames; ++targetInd) {
          if (hostInd == targetInd)
            continue;
          for (int targetCamInd = 0; targetCamInd < cam->bundle.size();
               ++targetCamInd) {
            SE3t &hostToTargetImage =
                hostToTarget[hostInd][hostCamInd][targetInd][targetCamInd];
            Vec3t rayTarget = hostToTargetImage * ray;
            CameraModel &camTarget = cam->bundle[targetCamInd].cam;
            if (!camTarget.isMappable(rayTarget))
              continue;
            Vec2t pointTarget = camTarget.map(rayTarget);
            if (!camTarget.isOnImage(pointTarget.cast<double>(), PH))
              continue;

            if (!hasResiduals) {
              hasResiduals = true;
              optimizationParams.logDepths.push_back(op.logDepth);
              points.push_back(&op);
              pointInd++;
            }

            residuals.emplace_back(
                hostInd, hostCamInd, targetInd, targetCamInd, pointInd - 1,
                &optimizationParams.logDepths.back(), cam,
                &keyFrames[hostInd]->frames[hostCamInd],
                &keyFrames[targetInd]->frames[targetCamInd], &op,
                hostToTargetImage, lossFunction.get(), settings);
          }
        }
      }
    }

  LOG(INFO) << "Created EnergyFunction with " << residuals.size()
            << " residuals\n";
}

SE3t EnergyFunction::getBodyToWorld(int frameInd) const {
  CHECK_GE(frameInd, 0);
  CHECK_LT(frameInd, keyFrames.size());
  if (frameInd == 0)
    return keyFrames[0]->thisToWorld().cast<T>();
  if (frameInd == 1)
    return optimizationParams.secondFrame.value();
  return optimizationParams.restFrames[frameInd - 2].value();
}

const MotionDerivatives &EnergyFunction::getHostToTargetDiff(int hostInd,
                                                             int hostCamInd,
                                                             int targetInd,
                                                             int targetCamInd) {
  std::optional<MotionDerivatives> &derivatives =
      hostToTargetDiff[hostInd][hostCamInd][targetInd][targetCamInd];
  if (derivatives)
    return derivatives.value();
  derivatives.emplace(cam->bundle[hostCamInd].thisToBody.cast<T>(),
                      getBodyToWorld(hostInd), getBodyToWorld(targetInd),
                      cam->bundle[targetCamInd].bodyToThis.cast<T>());
  return derivatives.value();
}

AffLightT EnergyFunction::getLightWorldToFrame(int frameInd, int frameCamInd) {
  if (frameInd == 0)
    return keyFrames[0]->frames[frameCamInd].lightWorldToThis.cast<T>();
  else
    return optimizationParams.lightWorldToFrame[frameInd - 1][frameCamInd];
}

const AffLightT &EnergyFunction::getLightHostToTarget(int hostInd,
                                                      int hostCamInd,
                                                      int targetInd,
                                                      int targetCamInd) {
  std::optional<AffLightT> &result =
      lightHostToTarget[hostInd][hostCamInd][targetInd][targetCamInd];
  if (result)
    return result.value();
  result.emplace(getLightWorldToFrame(targetInd, targetCamInd) *
                 getLightWorldToFrame(hostInd, hostCamInd).inverse());
  return result.value();
}

void EnergyFunction::recomputeHostToTarget() {
  for (int hostInd = 0; hostInd < keyFrames.size(); ++hostInd) {
    for (int targetInd = 0; targetInd < keyFrames.size(); ++targetInd) {
      if (hostInd == targetInd)
        continue;
      SE3t hostBodyToTargetBody =
          getBodyToWorld(targetInd).inverse() * getBodyToWorld(hostInd);
      for (int hostCamInd = 0; hostCamInd < cam->bundle.size(); ++hostCamInd) {
        SE3t hostFrameToTargetBody =
            hostBodyToTargetBody * cam->bundle[hostCamInd].thisToBody.cast<T>();
        for (int targetCamInd = 0; targetCamInd < cam->bundle.size();
             ++targetCamInd) {
          hostToTarget[hostInd][hostCamInd][targetInd][targetCamInd] =
              cam->bundle[targetCamInd].bodyToThis.cast<T>() *
              hostFrameToTargetBody;
        }
      }
    }
  }
}

void EnergyFunction::resetLightHostToTarget() {
  for (int hostInd = 0; hostInd < keyFrames.size(); ++hostInd)
    for (int targetInd = 0; targetInd < keyFrames.size(); ++targetInd)
      for (int hostCamInd = 0; hostCamInd < cam->bundle.size(); ++hostCamInd)
        for (int targetCamInd = 0; targetCamInd < cam->bundle.size();
             ++targetCamInd) {
          hostToTargetDiff[hostInd][hostCamInd][targetInd][targetCamInd]
              .reset();
          lightHostToTarget[hostInd][hostCamInd][targetInd][targetCamInd]
              .reset();
        }
}

void EnergyFunction::resetPrecomputations() {
  recomputeHostToTarget();
  resetLightHostToTarget();
}

EnergyFunction::Hessian EnergyFunction::getHessian() {
  constexpr int sndDoF = SecondFrameParametrization::DoF;
  constexpr int restDoF = FrameParametrization::DoF;
  constexpr int affDoF = AffLightT::DoF;
  constexpr int sndFrameDoF = sndDoF + affDoF, restFrameDoF = restDoF + affDoF;

  int nonconstFrames = keyFrames.size() - 1;
  int nonconstPoints = points.size();
  int framePars = (nonconstFrames - 1) * restFrameDoF + sndFrameDoF;
  int pointPars = nonconstPoints;

  Hessian hessian;
  hessian.frameFrame = MatXXt::Zero(framePars, framePars);
  hessian.framePoint = MatXXt::Zero(framePars, pointPars);
  hessian.pointPoint = VecXt::Zero(pointPars);

  Array2d<Accumulator<Residual::FrameFrameHessian>> frameFrameBlocks(
      boost::extents[nonconstFrames][nonconstFrames]);
  Array2d<Accumulator<Residual::FramePointHessian>> framePointBlocks(
      boost::extents[nonconstFrames][nonconstPoints]);

  for (Residual &residual : residuals) {
    int hi = residual.hostInd(), hci = residual.hostCamInd(),
        ti = residual.targetInd(), tci = residual.targetCamInd(),
        pi = residual.pointInd();
    CHECK_NE(hi, ti);

    const SE3t &curHostToTarget = hostToTarget[hi][hci][ti][tci];
    const MotionDerivatives &curDHostToTarget =
        getHostToTargetDiff(hi, hci, ti, tci);
    AffLightT lightWorldToHost = getLightWorldToFrame(hi, hci);
    AffLightT curLightHostToTarget = getLightHostToTarget(hi, hci, ti, tci);
    auto values = residual.getValues(curHostToTarget, curLightHostToTarget);
    Residual::Jacobian jacobian =
        residual.getJacobian(curHostToTarget, curDHostToTarget,
                             lightWorldToHost, curLightHostToTarget);
    Residual::DeltaHessian deltaHessian =
        residual.getDeltaHessian(values, jacobian);

    int him1 = hi - 1, tim1 = ti - 1;
    if (hi > 0)
      frameFrameBlocks[him1][him1] += deltaHessian.hostHost;
    if (ti > 0)
      frameFrameBlocks[tim1][tim1] += deltaHessian.targetTarget;
    if (hi > 0 && ti > 0) {
      if (hi < ti)
        frameFrameBlocks[him1][tim1] += deltaHessian.hostTarget;
      else
        frameFrameBlocks[tim1][him1] += deltaHessian.hostTarget.transpose();
    }

    if (hi > 0)
      framePointBlocks[him1][pi] += deltaHessian.hostPoint;
    if (ti > 0)
      framePointBlocks[tim1][pi] += deltaHessian.targetPoint;

    hessian.pointPoint[pi] += deltaHessian.pointPoint;
  }

  Mat75t sndParamDiff = optimizationParams.secondFrame.diffPlus();
  StdVector<Mat76t> restParamDiff;
  restParamDiff.reserve(nonconstFrames - 1);
  for (int i = 0; i < optimizationParams.restFrames.size(); ++i)
    restParamDiff.emplace_back(optimizationParams.restFrames[i].diffPlus());

  if (frameFrameBlocks[0][0].wasUsed()) {
    const Residual::FrameFrameHessian &topLeftBlock =
        frameFrameBlocks[0][0].accumulated();
    hessian.frameFrame.topLeftCorner<sndDoF, sndDoF>() =
        sndParamDiff.transpose() * topLeftBlock.qtqt * sndParamDiff;
    hessian.frameFrame.block<sndDoF, affDoF>(0, sndDoF) =
        sndParamDiff.transpose() * topLeftBlock.qtab;
    hessian.frameFrame.block<affDoF, affDoF>(sndDoF, sndDoF) =
        topLeftBlock.abab;
    hessian.frameFrame.block<affDoF, sndDoF>(sndDoF, 0) =
        hessian.frameFrame.block<sndDoF, affDoF>(0, sndDoF).transpose();
  }

  for (int i2 = 1, startCol = sndFrameDoF; i2 < nonconstFrames;
       ++i2, startCol += restFrameDoF) {
    if (!frameFrameBlocks[0][i2].wasUsed())
      continue;
    int i2m1 = i2 - 1;
    const Residual::FrameFrameHessian &curBlock =
        frameFrameBlocks[0][i2].accumulated();
    hessian.frameFrame.block<sndDoF, restDoF>(0, startCol) =
        sndParamDiff.transpose() * curBlock.qtqt * restParamDiff[i2m1];
    hessian.frameFrame.block<sndDoF, affDoF>(0, startCol + restDoF) =
        sndParamDiff.transpose() * curBlock.qtab;
    hessian.frameFrame.block<affDoF, restDoF>(sndDoF, startCol) =
        curBlock.abqt * restParamDiff[i2m1];
    hessian.frameFrame.block<affDoF, affDoF>(sndDoF, startCol + restDoF) =
        curBlock.abab;
  }
  for (int i1 = 1, startRow = sndFrameDoF; i1 < nonconstFrames;
       ++i1, startRow += restFrameDoF)
    for (int i2 = i1, startCol = startRow; i2 < nonconstFrames;
         ++i2, startCol += restFrameDoF) {
      if (!frameFrameBlocks[i1][i2].wasUsed())
        continue;
      int i1m1 = i1 - 1, i2m1 = i2 - 1;
      const Residual::FrameFrameHessian &curBlock =
          frameFrameBlocks[i1][i2].accumulated();

      hessian.frameFrame.block<restDoF, restDoF>(startRow, startCol) =
          restParamDiff[i1m1].transpose() * curBlock.qtqt * restParamDiff[i2m1];
      hessian.frameFrame.block<restDoF, affDoF>(startRow, startCol + restDoF) =
          restParamDiff[i1m1].transpose() * curBlock.qtab;
      hessian.frameFrame.block<affDoF, restDoF>(startRow + restDoF, startCol) =
          curBlock.abqt * restParamDiff[i2m1];
      hessian.frameFrame.block<affDoF, affDoF>(
          startRow + restDoF, startCol + restDoF) = curBlock.abab;
    }

  for (int pi = 0; pi < nonconstPoints; ++pi) {
    if (!framePointBlocks[0][pi].wasUsed())
      continue;
    const Residual::FramePointHessian &curBlock =
        framePointBlocks[0][pi].accumulated();
    hessian.framePoint.block<sndDoF, 1>(0, pi) =
        sndParamDiff.transpose() * curBlock.qtd;
    hessian.framePoint.block<affDoF, 1>(sndDoF, pi) = curBlock.abd;
  }

  for (int fi = 1, startRow = sndFrameDoF; fi < nonconstFrames;
       ++fi, startRow += restFrameDoF) {
    int fim1 = fi - 1;
    for (int pi = 0; pi < nonconstPoints; ++pi) {
      if (!framePointBlocks[fi][pi].wasUsed())
        continue;
      const Residual::FramePointHessian &curBlock =
          framePointBlocks[fi][pi].accumulated();

      hessian.framePoint.block<restDoF, 1>(startRow, pi) =
          restParamDiff[fim1].transpose() * curBlock.qtd;
      hessian.framePoint.block<affDoF, 1>(startRow + restDoF, pi) =
          curBlock.abd;
    }
  }

  {
    MatXXt &frameFrame = hessian.frameFrame;
    int rows = frameFrame.rows();
    for (int startRow = sndFrameDoF; startRow < rows;
         startRow += restFrameDoF) {
      int startCol = startRow == sndFrameDoF ? 0 : startRow - restFrameDoF;
      int blockRows = rows - startRow;
      int blockCols = startRow == 0 ? sndFrameDoF : restFrameDoF;
      frameFrame.block(startRow, startCol, blockRows, blockCols) =
          frameFrame.block(startCol, startRow, blockCols, blockRows)
              .transpose();
    }
  }

  int totalActive = 0;
  for (int fim1 = 0; fim1 < nonconstFrames; ++fim1)
    for (int pi = 0; pi < nonconstPoints; ++pi)
      if (framePointBlocks[fim1][pi].wasUsed())
        totalActive++;

  int totalBlocks = nonconstFrames * nonconstPoints;
  LOG(INFO) << "Computed Hessian. nonconst frames = " << nonconstFrames
            << ", nonconst points = " << nonconstPoints
            << ", total blocks = " << totalBlocks
            << ", active blocks = " << totalActive
            << ", fill factor = " << double(totalActive) / totalBlocks * 100
            << "%";

  return hessian;
}

} // namespace mdso::optimize