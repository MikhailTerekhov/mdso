#include "optimize/EnergyFunction.h"
#include "optimize/Accumulator.h"

namespace mdso::optimize {

EnergyFunction::EnergyFunction(CameraBundle *camBundle, KeyFrame **keyFrames,
                               int numKeyFrames,
                               const ResidualSettings &settings)
    : parameters(camBundle, keyFrames, numKeyFrames)
    , lossFunction(new ceres::HuberLoss(settings.intensity.outlierDiff))
    , cam(camBundle)
    , settings(settings) {
  CHECK(numKeyFrames >= 2);

  int PH = settings.residualPattern.height;

  PrecomputedHostToTarget hostToTarget(cam, &parameters);
  std::vector<OptimizedPoint *> optimizedPoints;

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
            SE3t hostToTargetImage =
                hostToTarget.get(hostInd, hostCamInd, targetInd, targetCamInd);
            Vec3t rayTarget = hostToTargetImage * ray;
            CameraModel &camTarget = cam->bundle[targetCamInd].cam;
            if (!camTarget.isMappable(rayTarget))
              continue;
            Vec2t pointTarget = camTarget.map(rayTarget);
            if (!camTarget.isOnImage(pointTarget.cast<double>(), PH))
              continue;

            if (!hasResiduals) {
              hasResiduals = true;
              optimizedPoints.push_back(&op);
            }

            residuals.emplace_back(hostInd, hostCamInd, targetInd, targetCamInd,
                                   optimizedPoints.size() - 1, cam,
                                   &keyFrames[hostInd]->frames[hostCamInd],
                                   &keyFrames[targetInd]->frames[targetCamInd],
                                   &op, op.logDepth, hostToTargetImage,
                                   lossFunction.get(), settings);
          }
        }
      }
    }

  parameters.setPoints(std::move(optimizedPoints));

  LOG(INFO) << "Created EnergyFunction with " << residuals.size()
            << " residuals\n";
}

int EnergyFunction::numPoints() const { return parameters.numPoints(); }

VecRt EnergyFunction::getResidualValues(int residualInd) {
  CHECK_GE(residualInd, 0);
  CHECK_LT(residualInd, residuals.size());
  return getAllValues().values(residualInd);
}

EnergyFunction::Hessian EnergyFunction::getHessian() {
  PrecomputedHostToTarget hostToTarget(cam, &parameters);
  PrecomputedMotionDerivatives motionDerivatives(cam, &parameters);
  PrecomputedLightHostToTarget lightHostToTarget(&parameters);
  return getHessian(hostToTarget, motionDerivatives, lightHostToTarget);
}

EnergyFunction::Hessian
EnergyFunction::getHessian(PrecomputedHostToTarget &hostToTarget,
                           PrecomputedMotionDerivatives &motionDerivatives,
                           PrecomputedLightHostToTarget &lightHostToTarget) {
  constexpr int sndDoF = Parameters::sndDoF;
  constexpr int restDoF = Parameters::restDoF;
  constexpr int affDoF = Parameters::affDoF;
  constexpr int sndFrameDoF = Parameters::sndFrameDoF;
  constexpr int restFrameDoF = Parameters::restFrameDoF;

  int nonconstFrames = parameters.numKeyFrames() - 1;
  int nonconstPoints = parameters.numPoints();
  int framePars = parameters.frameParameters();
  int pointPars = nonconstPoints;

  const Values &valuesRef = getAllValues(hostToTarget, lightHostToTarget);
  const Derivatives &derivativesRef =
      getDerivatives(hostToTarget, motionDerivatives, lightHostToTarget);

  Hessian hessian;
  hessian.frameFrame = MatXXt::Zero(framePars, framePars);
  hessian.framePoint = MatXXt::Zero(framePars, pointPars);
  hessian.pointPoint = VecXt::Zero(pointPars);

  Array2d<Accumulator<Residual::FrameFrameHessian>> frameFrameBlocks(
      boost::extents[nonconstFrames][nonconstFrames]);
  Array2d<Accumulator<Residual::FramePointHessian>> framePointBlocks(
      boost::extents[nonconstFrames][nonconstPoints]);

  for (int i = 0; i < residuals.size(); ++i) {
    Residual &residual = residuals[i];
    int hi = residual.hostInd(), hci = residual.hostCamInd(),
        ti = residual.targetInd(), tci = residual.targetCamInd(),
        pi = residual.pointInd();
    CHECK_NE(hi, ti);

    Residual::DeltaHessian deltaHessian = residual.getDeltaHessian(
        getResidualValues(i), derivativesRef.residualJacobians[i]);

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

  const Mat75t &sndParamDiff =
      derivativesRef.parametrizationJacobians.dSecondFrame;

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
    const Mat76t &paramDiff =
        derivativesRef.parametrizationJacobians.dRestFrames[i2m1];
    const Residual::FrameFrameHessian &curBlock =
        frameFrameBlocks[0][i2].accumulated();
    hessian.frameFrame.block<sndDoF, restDoF>(0, startCol) =
        sndParamDiff.transpose() * curBlock.qtqt * paramDiff;
    hessian.frameFrame.block<sndDoF, affDoF>(0, startCol + restDoF) =
        sndParamDiff.transpose() * curBlock.qtab;
    hessian.frameFrame.block<affDoF, restDoF>(sndDoF, startCol) =
        curBlock.abqt * paramDiff;
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
      const Mat76t &hostParamDiff =
          derivativesRef.parametrizationJacobians.dRestFrames[i1m1];
      const Mat76t &targetParamDiff =
          derivativesRef.parametrizationJacobians.dRestFrames[i2m1];

      hessian.frameFrame.block<restDoF, restDoF>(startRow, startCol) =
          hostParamDiff.transpose() * curBlock.qtqt * targetParamDiff;
      hessian.frameFrame.block<restDoF, affDoF>(startRow, startCol + restDoF) =
          hostParamDiff.transpose() * curBlock.qtab;
      hessian.frameFrame.block<affDoF, restDoF>(startRow + restDoF, startCol) =
          curBlock.abqt * targetParamDiff;
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
      const Mat76t &paramDiff =
          derivativesRef.parametrizationJacobians.dRestFrames[fim1];

      hessian.framePoint.block<restDoF, 1>(startRow, pi) =
          paramDiff.transpose() * curBlock.qtd;
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

EnergyFunction::Gradient EnergyFunction::getGradient() {
  PrecomputedHostToTarget hostToTarget(cam, &parameters);
  PrecomputedMotionDerivatives motionDerivatives(cam, &parameters);
  PrecomputedLightHostToTarget lightHostToTarget(&parameters);
  return getGradient(hostToTarget, motionDerivatives, lightHostToTarget);
}

EnergyFunction::Gradient
EnergyFunction::getGradient(PrecomputedHostToTarget &hostToTarget,
                            PrecomputedMotionDerivatives &motionDerivatives,
                            PrecomputedLightHostToTarget &lightHostToTarget) {
  constexpr int sndDoF = Parameters::sndDoF;
  constexpr int restDoF = Parameters::restDoF;
  constexpr int affDoF = Parameters::affDoF;
  constexpr int sndFrameDoF = Parameters::sndFrameDoF;
  constexpr int restFrameDoF = Parameters::restFrameDoF;

  int nonconstFrames = parameters.numKeyFrames() - 1;
  int nonconstPoints = parameters.numPoints();
  int framePars = parameters.frameParameters();
  int pointPars = nonconstPoints;

  const Values &valuesRef = getAllValues(hostToTarget, lightHostToTarget);
  const Derivatives &derivativesRef =
      getDerivatives(hostToTarget, motionDerivatives, lightHostToTarget);

  EnergyFunction::Gradient gradient;
  gradient.frame = VecXt::Zero(parameters.frameParameters());
  gradient.point = VecXt::Zero(parameters.numPoints());

  std::vector<Accumulator<Residual::FrameGradient>> frameBlocks(nonconstFrames);

  for (int i = 0; i < residuals.size(); ++i) {
    Residual &residual = residuals[i];
    int hi = residual.hostInd(), hci = residual.hostCamInd(),
        ti = residual.targetInd(), tci = residual.targetCamInd(),
        pi = residual.pointInd();
    int him1 = hi - 1, tim1 = ti - 1;
    CHECK_NE(hi, ti);

    Residual::DeltaGradient deltaGradient = residual.getDeltaGradient(
        valuesRef.values(i), derivativesRef.residualJacobians[i]);
    if (hi > 0)
      frameBlocks[him1] += deltaGradient.host;
    if (ti > 0)
      frameBlocks[tim1] += deltaGradient.target;
    gradient.point[pi] += deltaGradient.point;
  }

  const Mat75t &sndParam = derivativesRef.parametrizationJacobians.dSecondFrame;
  const auto &restParams = derivativesRef.parametrizationJacobians.dRestFrames;
  if (frameBlocks[0].wasUsed()) {
    const Residual::FrameGradient &sndFrameGradient =
        frameBlocks[0].accumulated();
    gradient.frame.head<sndDoF>() = sndParam.transpose() * sndFrameGradient.qt;
    gradient.frame.segment<affDoF>(sndDoF) = sndFrameGradient.ab;
  }
  for (int fi = 1, startElem = sndFrameDoF; fi < nonconstFrames;
       ++fi, startElem += restFrameDoF)
    if (frameBlocks[fi].wasUsed()) {
      int fim1 = fi - 1;
      const Residual::FrameGradient &frameBlock = frameBlocks[fi].accumulated();
      gradient.frame.segment<restDoF>(startElem) =
          restParams[fim1].transpose() * frameBlock.qt;
      gradient.frame.segment<affDoF>(startElem + restDoF) = frameBlock.ab;
    }

  return gradient;
}

void EnergyFunction::precomputeValuesAndDerivatives() {
  PrecomputedHostToTarget hostToTarget(cam, &parameters);
  PrecomputedMotionDerivatives motionDerivatives(cam, &parameters);
  PrecomputedLightHostToTarget lightHostToTarget(&parameters);
  getAllValues(hostToTarget, lightHostToTarget);
  getDerivatives(hostToTarget, motionDerivatives, lightHostToTarget);
}

void EnergyFunction::clearPrecomputations() {
  values.reset();
  derivatives.reset();
}

EnergyFunction::Parameters::Jacobians::Jacobians(const State &state)
    : dSecondFrame(state.secondFrame.diffPlus())
    , dRestFrames(state.restFrames.size()) {
  for (int i = 0; i < dRestFrames.size(); ++i)
    dRestFrames[i] = state.restFrames[i].diffPlus();
}

EnergyFunction::Parameters::Jacobians::Jacobians(const Parameters &parameters)
    : Jacobians(parameters.state) {}

EnergyFunction::Parameters::State::State(KeyFrame **keyFrames,
                                         int newNumKeyFrames)
    : firstBodyToWorld(keyFrames[0]->thisToWorld().cast<T>())
    , secondFrame(keyFrames[0]->thisToWorld(), keyFrames[1]->thisToWorld())
    , lightWorldToFrame(
          boost::extents[newNumKeyFrames]
                        [keyFrames[0]->preKeyFrame->cam->bundle.size()]) {
  CHECK_GE(newNumKeyFrames, 2);
  int bundleSize = keyFrames[0]->preKeyFrame->cam->bundle.size();

  restFrames.reserve(newNumKeyFrames - 2);
  for (int i = 2; i < newNumKeyFrames; ++i)
    restFrames.emplace_back(keyFrames[i]->thisToWorld().cast<T>());

  for (int fi = 0; fi < newNumKeyFrames; ++fi)
    for (int ci = 0; ci < bundleSize; ++ci)
      lightWorldToFrame[fi][ci] =
          keyFrames[fi]->frames[ci].lightWorldToThis.cast<T>();
}

int EnergyFunction::Parameters::State::frameParameters() const {
  return restFrames.size() * restFrameDoF + sndFrameDoF;
}

void EnergyFunction::Parameters::State::applyUpdate(const VecX &deltaFrame,
                                                    const VecX &deltaPoints) {
  CHECK_EQ(deltaFrame.size(), frameParameters());
  CHECK_EQ(deltaPoints.size(), logDepths.size());
}

EnergyFunction::Parameters::Parameters(CameraBundle *cam,
                                       KeyFrame **newKeyFrames,
                                       int newNumKeyFrames)
    : state(newKeyFrames, newNumKeyFrames)
    , keyFrames(newKeyFrames, newKeyFrames + newNumKeyFrames) {}

SE3t EnergyFunction::Parameters::getBodyToWorld(int frameInd) const {
  CHECK_GE(frameInd, 0);
  CHECK_LT(frameInd, state.restFrames.size() + 2);
  if (frameInd == 0)
    return state.firstBodyToWorld;
  if (frameInd == 1)
    return state.secondFrame.value();
  return state.restFrames[frameInd - 2].value();
}

AffLightT
EnergyFunction::Parameters::getLightWorldToFrame(int frameInd,
                                                 int frameCamInd) const {
  return state.lightWorldToFrame[frameInd][frameCamInd];
}

int EnergyFunction::Parameters::numKeyFrames() const {
  return state.restFrames.size() + 2;
}

int EnergyFunction::Parameters::numPoints() const {
  return state.logDepths.size();
}

int EnergyFunction::Parameters::camBundleSize() const {
  return state.lightWorldToFrame.shape()[1];
}

int EnergyFunction::Parameters::frameParameters() const {
  return state.frameParameters();
}

T EnergyFunction::Parameters::logDepth(int i) const {
  CHECK_GE(i, 0);
  CHECK_LT(i, state.logDepths.size());
  return state.logDepths[i];
}

void EnergyFunction::Parameters::setPoints(
    std::vector<OptimizedPoint *> &&newOptimizedPoints) {
  optimizedPoints = newOptimizedPoints;
  state.logDepths = VecXt(optimizedPoints.size());
  for (int i = 0; i < optimizedPoints.size(); ++i)
    state.logDepths[i] = optimizedPoints[i]->logDepth;
}

EnergyFunction::Parameters::State
EnergyFunction::Parameters::saveState() const {
  return state;
}

void EnergyFunction::Parameters::recoverState(
    optimize::EnergyFunction::Parameters::State &&oldState) {
  state = oldState;
  CHECK_EQ(state.restFrames.size() + 2, keyFrames.size());
  CHECK_EQ(state.logDepths.size(), optimizedPoints.size());
}

void EnergyFunction::Parameters::update(const VecX &deltaFrame,
                                        const VecX &deltaPoints) {
  state.applyUpdate(deltaFrame, deltaPoints);
}

void EnergyFunction::Parameters::apply() {
  keyFrames[1]->thisToWorld.setValue(state.secondFrame.value().cast<double>());
  for (int fi = 0; fi < state.restFrames.size(); ++fi)
    keyFrames[fi + 2]->thisToWorld.setValue(
        state.restFrames[fi].value().cast<double>());
  for (int pi = 0; pi < optimizedPoints.size(); ++pi)
    optimizedPoints[pi]->logDepth = state.logDepths[pi];
}

EnergyFunction::Values::Values(
    const StdVector<Residual> &residuals, const Parameters &parameters,
    PrecomputedHostToTarget &hostToTarget,
    PrecomputedLightHostToTarget &lightHostToTarget) {
  valsAndCache.reserve(residuals.size());
  for (const Residual &res : residuals) {
    int hi = res.hostInd(), hci = res.hostCamInd(), ti = res.targetInd(),
        tci = res.targetCamInd(), pi = res.pointInd();
    valsAndCache.push_back(
        {VecRt(res.patternSize()), Residual::CachedValues(res.patternSize())});
    valsAndCache.back().first =
        res.getValues(hostToTarget.get(hi, hci, ti, tci),
                      lightHostToTarget.get(hi, hci, ti, tci),
                      parameters.logDepth(pi), &valsAndCache.back().second);
  }
}

const VecRt &EnergyFunction::Values::values(int residualInd) const {
  CHECK_GE(residualInd, 0);
  CHECK_LT(residualInd, valsAndCache.size());
  return valsAndCache[residualInd].first;
}

const Residual::CachedValues &
EnergyFunction::Values::cachedValues(int residualInd) const {
  CHECK_GE(residualInd, 0);
  CHECK_LT(residualInd, valsAndCache.size());
  return valsAndCache[residualInd].second;
}

EnergyFunction::PrecomputedHostToTarget::PrecomputedHostToTarget(
    CameraBundle *cam, const Parameters *parameters)
    : parameters(parameters)
    , camToBody(cam->bundle.size())
    , bodyToCam(cam->bundle.size())
    , hostToTarget(
          boost::extents[parameters->numKeyFrames()][cam->bundle.size()]
                        [parameters->numKeyFrames()][cam->bundle.size()]) {
  for (int ci = 0; ci < cam->bundle.size(); ++ci) {
    camToBody[ci] = cam->bundle[ci].thisToBody.cast<T>();
    bodyToCam[ci] = cam->bundle[ci].bodyToThis.cast<T>();
  }

  for (int hostInd = 0; hostInd < parameters->numKeyFrames(); ++hostInd) {
    for (int targetInd = 0; targetInd < parameters->numKeyFrames();
         ++targetInd) {
      if (hostInd == targetInd)
        continue;
      SE3t hostBodyToTargetBody =
          parameters->getBodyToWorld(targetInd).inverse() *
          parameters->getBodyToWorld(hostInd);
      for (int hostCamInd = 0; hostCamInd < cam->bundle.size(); ++hostCamInd) {
        SE3t hostFrameToTargetBody =
            hostBodyToTargetBody * camToBody[hostCamInd];
        for (int targetCamInd = 0; targetCamInd < cam->bundle.size();
             ++targetCamInd) {
          hostToTarget[hostInd][hostCamInd][targetInd][targetCamInd] =
              bodyToCam[targetCamInd] * hostFrameToTargetBody;
        }
      }
    }
  }
}

SE3t EnergyFunction::PrecomputedHostToTarget::get(int hostInd, int hostCamInd,
                                                  int targetInd,
                                                  int targetCamInd) {
  return hostToTarget[hostInd][hostCamInd][targetInd][targetCamInd];
}

EnergyFunction::PrecomputedMotionDerivatives::PrecomputedMotionDerivatives(
    CameraBundle *cam, const Parameters *parameters)
    : parameters(parameters)
    , camToBody(cam->bundle.size())
    , bodyToCam(cam->bundle.size())
    , hostToTargetDiff(
          boost::extents[parameters->numKeyFrames()][cam->bundle.size()]
                        [parameters->numKeyFrames()][cam->bundle.size()]) {
  for (int ci = 0; ci < cam->bundle.size(); ++ci) {
    camToBody[ci] = cam->bundle[ci].thisToBody.cast<T>();
    bodyToCam[ci] = cam->bundle[ci].bodyToThis.cast<T>();
  }
}

const MotionDerivatives &EnergyFunction::PrecomputedMotionDerivatives::get(
    int hostInd, int hostCamInd, int targetInd, int targetCamInd) {
  std::optional<MotionDerivatives> &derivatives =
      hostToTargetDiff[hostInd][hostCamInd][targetInd][targetCamInd];
  if (derivatives)
    return derivatives.value();
  derivatives.emplace(
      camToBody[hostCamInd], parameters->getBodyToWorld(hostInd),
      parameters->getBodyToWorld(targetInd), bodyToCam[targetCamInd]);
  return derivatives.value();
}

EnergyFunction::PrecomputedLightHostToTarget::PrecomputedLightHostToTarget(
    const Parameters *parameters)
    : parameters(parameters)
    , lightHostToTarget(boost::extents[parameters->numKeyFrames()]
                                      [parameters->camBundleSize()]
                                      [parameters->numKeyFrames()]
                                      [parameters->camBundleSize()]) {}

AffLightT EnergyFunction::PrecomputedLightHostToTarget::get(int hostInd,
                                                            int hostCamInd,
                                                            int targetInd,
                                                            int targetCamInd) {
  std::optional<AffLightT> &result =
      lightHostToTarget[hostInd][hostCamInd][targetInd][targetCamInd];
  if (result)
    return result.value();
  result.emplace(
      parameters->getLightWorldToFrame(targetInd, targetCamInd) *
      parameters->getLightWorldToFrame(hostInd, hostCamInd).inverse());
  return result.value();
}

EnergyFunction::Derivatives::Derivatives(
    const Parameters &parameters, const StdVector<Residual> &residuals,
    const Values &values, PrecomputedHostToTarget &hostToTarget,
    PrecomputedMotionDerivatives &motionDerivatives,
    PrecomputedLightHostToTarget &lightHostToTarget)
    : parametrizationJacobians(parameters) {
  residualJacobians.reserve(residuals.size());
  for (int ri = 0; ri < residuals.size(); ++ri) {
    const Residual &res = residuals[ri];
    int hi = res.hostInd(), hci = res.hostCamInd(), ti = res.targetInd(),
        tci = res.targetCamInd(), pi = res.pointInd();
    SE3t curHostToTarget = hostToTarget.get(hi, hci, ti, tci);
    AffLightT curLightHostToTarget = lightHostToTarget.get(hi, hci, ti, tci);
    AffLightT lightWorldToHost = parameters.getLightWorldToFrame(hi, hci);
    const MotionDerivatives &dHostToTarget =
        motionDerivatives.get(hi, hci, ti, tci);

    T logDepth = parameters.logDepth(pi);
    residualJacobians.push_back(res.getJacobian(
        curHostToTarget, dHostToTarget, lightWorldToHost, curLightHostToTarget,
        logDepth, values.cachedValues(ri)));
  }
}

EnergyFunction::Values &EnergyFunction::getAllValues() {
  if (!values) {
    PrecomputedHostToTarget hostToTarget(cam, &parameters);
    PrecomputedLightHostToTarget lightHostToTarget(&parameters);
    return getAllValues(hostToTarget, lightHostToTarget);
  }
  return values.value();
}

EnergyFunction::Values &
EnergyFunction::getAllValues(PrecomputedHostToTarget &hostToTarget,
                             PrecomputedLightHostToTarget &lightHostToTarget) {
  if (!values) {
    values.emplace(residuals, parameters, hostToTarget, lightHostToTarget);
  }
  return values.value();
}

EnergyFunction::Derivatives &EnergyFunction::getDerivatives(
    PrecomputedHostToTarget &hostToTarget,
    PrecomputedMotionDerivatives &motionDerivatives,
    PrecomputedLightHostToTarget &lightHostToTarget) {
  if (!derivatives)
    derivatives.emplace(parameters, residuals,
                        getAllValues(hostToTarget, lightHostToTarget),
                        hostToTarget, motionDerivatives, lightHostToTarget);
  return derivatives.value();
}

} // namespace mdso::optimize