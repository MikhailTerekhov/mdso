#include "optimize/EnergyFunction.h"
#include "optimize/Accumulator.h"
#include "optimize/StepController.h"
#include "system/Reprojector.h"

namespace mdso::optimize {

std::unique_ptr<ceres::LossFunction> getLoss(Settings::Optimization::Loss type,
                                             double outlierDiff) {
  using loss_ptr = std::unique_ptr<ceres::LossFunction>;

  switch (type) {
  case Settings::Optimization::TRIVIAL:
    return loss_ptr(new ceres::TrivialLoss());
  case Settings::Optimization::HUBER:
    return loss_ptr(new ceres::HuberLoss(outlierDiff));
  default:
    return loss_ptr(new ceres::TrivialLoss());
  }
}

EnergyFunction::EnergyFunction(CameraBundle *camBundle, KeyFrame **keyFrames,
                               int numKeyFrames,
                               const EnergyFunctionSettings &settings)
    : parameters(new Parameters(camBundle, keyFrames, numKeyFrames))
    , lossFunction(getLoss(settings.optimization.lossType,
                           settings.residual.intensity.outlierDiff))
    , cam(camBundle)
    , settings(settings) {
  CHECK(numKeyFrames >= 2);

  int PH = settings.residual.residualPattern.height;

  PrecomputedHostToTarget hostToTarget(cam, parameters.get());
  std::vector<OptimizedPoint *> optimizedPoints;
  Array2d<std::vector<int>> globalPointInds(
      boost::extents[numKeyFrames][cam->bundle.size()]);
  for (int kfInd = 0; kfInd < numKeyFrames; ++kfInd)
    for (int camInd = 0; camInd < cam->bundle.size(); ++camInd)
      globalPointInds[kfInd][camInd].resize(
          keyFrames[kfInd]->frames[camInd].optimizedPoints.size(), -1);

  for (int targetInd = 0; targetInd < numKeyFrames; ++targetInd) {
    Reprojector<OptimizedPoint> reprojector(keyFrames, numKeyFrames,
                                            keyFrames[targetInd]->thisToWorld(),
                                            settings.residual.depth, PH);
    reprojector.setSkippedFrame(targetInd);
    StdVector<Reprojection> reprojections = reprojector.reproject();

    for (const auto &repr : reprojections) {
      OptimizedPoint &op = keyFrames[repr.hostInd]
                               ->frames[repr.hostCamInd]
                               .optimizedPoints[repr.pointInd];

      if ((settings.residual.depth.setMinBound ||
           settings.residual.depth.useMinPlusExpParametrization) &&
          op.depth() <= settings.residual.depth.min)
        continue;
      if (settings.residual.depth.setMaxBound &&
          op.depth() > settings.residual.depth.max)
        continue;
      if (op.depth() < 0 || !std::isfinite(op.depth()))
        continue;

      int &globalPointInd =
          globalPointInds[repr.hostInd][repr.hostCamInd][repr.pointInd];
      if (globalPointInd == -1) {
        globalPointInd = optimizedPoints.size();
        optimizedPoints.push_back(&op);
      }

      SE3t hostToTargetImage = hostToTarget.get(repr.hostInd, repr.hostCamInd,
                                                targetInd, repr.targetCamInd);

      residuals.emplace_back(repr.hostInd, repr.hostCamInd, targetInd,
                             repr.targetCamInd, globalPointInd, cam,
                             &keyFrames[repr.hostInd]->frames[repr.hostCamInd],
                             &keyFrames[targetInd]->frames[repr.targetCamInd],
                             &op, op.logDepth, hostToTargetImage,
                             lossFunction.get(), settings.residual);
    }
  }

  parameters->setPoints(optimizedPoints);

  LOG(INFO) << "Created EnergyFunction with " << residuals.size()
            << " residuals and " << optimizedPoints.size() << " points";
}

int EnergyFunction::numPoints() const { return parameters->numPoints(); }

VecRt EnergyFunction::getResidualValues(int residualInd) {
  CHECK_GE(residualInd, 0);
  CHECK_LT(residualInd, residuals.size());
  return computeValues().values(residualInd);
}

VecRt EnergyFunction::getPredictedResidualIncrement(
    int residualInd, const DeltaParameterVector &delta) {
  int patternSize = settings.residual.patternSize();
  const Derivatives &curDerivatives = computeDerivatives();
  auto getFrameDeltaR =
      [&](int frameInd, int frameCamInd, const MatR2t &gradITarget,
          const Residual::Jacobian::DiffFrameParams &diffFrameParams) -> VecRt {
    if (frameInd == 0)
      return VecRt::Zero(patternSize);

    MatR7t dr_dqt(patternSize, 7);
    dr_dqt << gradITarget * diffFrameParams.dp_dq,
        gradITarget * diffFrameParams.dp_dt;

    VecRt motionDeltaR(patternSize);
    if (frameInd == 1)
      motionDeltaR = dr_dqt *
                     curDerivatives.parametrizationJacobians.dSecondFrame() *
                     delta.sndBlock();
    else
      motionDeltaR =
          dr_dqt *
          curDerivatives.parametrizationJacobians.dOtherFrame(frameInd) *
          delta.restBlock(frameInd);

    VecRt affDeltaR =
        diffFrameParams.dr_dab * delta.affBlock(frameInd, frameCamInd);

    return motionDeltaR + affDeltaR;
  };

  const Residual &res = residuals[residualInd];
  int hi = res.hostInd(), hci = res.hostCamInd(), ti = res.targetInd(),
      tci = res.targetCamInd(), pi = res.pointInd();
  const Residual::Jacobian &jacobian =
      curDerivatives.residualJacobians[residualInd];
  VecRt hostDeltaR =
      getFrameDeltaR(hi, hci, jacobian.gradItarget, jacobian.dhost);
  VecRt targetDeltaR =
      getFrameDeltaR(ti, tci, jacobian.gradItarget, jacobian.dtarget);
  double deltaPoint = delta.pointBlock(pi);
  VecRt pointDeltaR(patternSize);
  if (deltaPoint == 0)
    pointDeltaR.setZero();
  else
    pointDeltaR = jacobian.dr_dlogd() * deltaPoint;
  VecRt gradWeights = res.getPixelDependentWeights();
  return gradWeights.cwiseSqrt().cwiseProduct(hostDeltaR + targetDeltaR +
                                              pointDeltaR);
}

Hessian EnergyFunction::getHessian() {
  PrecomputedHostToTarget hostToTarget(cam, parameters.get());
  PrecomputedMotionDerivatives motionDerivatives(cam, parameters.get());
  PrecomputedLightHostToTarget lightHostToTarget(parameters.get());
  const Values &values = computeValues(hostToTarget, lightHostToTarget);
  const Derivatives &derivatives =
      computeDerivatives(hostToTarget, motionDerivatives, lightHostToTarget);
  return getHessian(values, derivatives);
}

Hessian EnergyFunction::getHessian(const Values &precomputedValues,
                                   const Derivatives &precomputedDerivatives) {
  Hessian::AccumulatedBlocks accumulatedBlocks(parameters->numKeyFrames(),
                                               parameters->numCameras(),
                                               parameters->numPoints());
  for (int ri = 0; ri < residuals.size(); ++ri) {
    const Residual &residual = residuals[ri];
    Residual::DeltaHessian deltaHessian =
        residual.getDeltaHessian(precomputedValues.values(ri),
                                 precomputedDerivatives.residualJacobians[ri]);
    accumulatedBlocks.add(residual, deltaHessian);
  }

  return Hessian(accumulatedBlocks,
                 precomputedDerivatives.parametrizationJacobians,
                 settings.optimization);
}

Gradient EnergyFunction::getGradient() {
  PrecomputedHostToTarget hostToTarget(cam, parameters.get());
  PrecomputedMotionDerivatives motionDerivatives(cam, parameters.get());
  PrecomputedLightHostToTarget lightHostToTarget(parameters.get());
  const Values &valuesRef = computeValues(hostToTarget, lightHostToTarget);
  const Derivatives &derivativesRef =
      computeDerivatives(hostToTarget, motionDerivatives, lightHostToTarget);
  return getGradient(valuesRef, derivativesRef);
}

Gradient
EnergyFunction::getGradient(const Values &precomputedValues,
                            const Derivatives &precomputedDerivatives) {
  Gradient::AccumulatedBlocks accumulatedBlocks(parameters->numKeyFrames(),
                                                parameters->numCameras(),
                                                parameters->numPoints());

  for (int ri = 0; ri < residuals.size(); ++ri) {
    const Residual &residual = residuals[ri];
    Residual::DeltaGradient deltaGradient =
        residual.getDeltaGradient(precomputedValues.values(ri),
                                  precomputedDerivatives.residualJacobians[ri]);
    accumulatedBlocks.add(residual, deltaGradient);
  }

  return Gradient(accumulatedBlocks,
                  precomputedDerivatives.parametrizationJacobians);
}

void EnergyFunction::precomputeValuesAndDerivatives() {
  PrecomputedHostToTarget hostToTarget(cam, parameters.get());
  PrecomputedMotionDerivatives motionDerivatives(cam, parameters.get());
  PrecomputedLightHostToTarget lightHostToTarget(parameters.get());
  computeValues(hostToTarget, lightHostToTarget);
  computeDerivatives(hostToTarget, motionDerivatives, lightHostToTarget);
}

void EnergyFunction::clearPrecomputations() {
  values.reset();
  derivatives.reset();
}

Parameters::State EnergyFunction::saveState() const {
  return parameters->saveState();
}

void EnergyFunction::recoverState(const Parameters::State &oldState) {
  parameters->recoverState(oldState);
  clearPrecomputations();
}

std::vector<int> oobDepthInds(const VecXt &logDepths, T minLogDepth,
                              T maxLogDepth) {
  std::vector<int> oobInds;
  for (int i = 0; i < logDepths.size(); ++i)
    if (logDepths[i] < minLogDepth || logDepths[i] > maxLogDepth)
      oobInds.push_back(i);
  return oobInds;
}

void EnergyFunction::optimize(int maxIterations) {
  T minLogDepth = std::log(settings.depth.min);
  T maxLogDepth = std::log(settings.depth.max);

  auto hostToTarget = precomputeHostToTarget();
  auto motionDerivatives = precomputeMotionDerivatives();
  auto lightHostToTarget = precomputeLightHostToTarget();
  Values &curValues = computeValues(hostToTarget, lightHostToTarget);
  Derivatives &curDerivatives =
      computeDerivatives(hostToTarget, motionDerivatives, lightHostToTarget);
  Hessian hessian = getHessian(curValues, curDerivatives);
  Gradient gradient = getGradient(curValues, curDerivatives);
  StepController stepController(settings.optimization.stepControl);
  bool parametersUpdated = false;
  int numSuccessfulIterations = 0, consecutiveFailedIterations = 0;
  int curIteration = 0;
  while (numSuccessfulIterations < maxIterations &&
         consecutiveFailedIterations <
             settings.optimization.maxConsecutiveFailedIterations) {
    LOG(INFO) << "it = " << curIteration << "\n";
    TimePoint start, end;
    start = now();

    double curEnergy = curValues.totalEnergy(residuals);
    LOG(INFO) << "cur energy = " << curEnergy << "\n";

    if (parametersUpdated) {
      motionDerivatives = precomputeMotionDerivatives();
      curDerivatives = createDerivatives(curValues, hostToTarget,
                                         motionDerivatives, lightHostToTarget);
      hessian = getHessian(curValues, curDerivatives);
      gradient = getGradient(curValues, curDerivatives);
    }

    Hessian dampedHessian =
        hessian.levenbergMarquardtDamp(stepController.lambda());

    Parameters::State savedState = parameters->saveState();
    std::vector<int> oobInds =
        oobDepthInds(savedState.logDepths, minLogDepth, maxLogDepth);
    LOG(INFO) << "OOB depths count = " << oobInds.size()
              << " (these were excluded from the current optimization step)";

    DeltaParameterVector delta =
        dampedHessian.solve(gradient, oobInds.data(), oobInds.size());
    if (!settings.affineLight.optimizeAffineLight)
      delta.setAffineZero();
    delta.constraintDepths(settings.depth.maxAbsLogDelta);

    LOG(INFO) << "delta : frame norm = " << delta.getFrame().norm()
              << " point norm = " << delta.getPoint().norm();

    double predictedEnergy =
        curEnergy + getPredictedDeltaEnergy(hessian, gradient, delta);
    double predictedViaJacobian = predictEnergyViaJacobian(delta);
    LOG(INFO) << "predicted via J = " << predictedViaJacobian
              << ", diff = " << curEnergy - predictedViaJacobian;

    parameters->update(delta);

    auto newHostToTarget = precomputeHostToTarget();
    auto newLightHostToTarget = precomputeLightHostToTarget();
    Values newValues = createValues(newHostToTarget, newLightHostToTarget);

    double newEnergy = newValues.totalEnergy(residuals);

    LOG(INFO) << "optimization step #" << curIteration
              << ": curEnergy = " << curEnergy
              << " predictedEnergy = " << predictedEnergy
              << " newEnergy = " << newEnergy
              << " delta = " << newEnergy - curEnergy;

    parametersUpdated =
        stepController.newStep(curEnergy, newEnergy, predictedEnergy);
    if (parametersUpdated) {
      curValues = std::move(newValues);
      hostToTarget = std::move(newHostToTarget);
      lightHostToTarget = std::move(newLightHostToTarget);
      numSuccessfulIterations++;
      consecutiveFailedIterations = 0;
    } else {
      parameters->recoverState(std::move(savedState));
      consecutiveFailedIterations++;
    }
    curIteration++;

    end = now();
    LOG(INFO) << "step took " << secondsBetween(start, end);
  }

  parameters->apply();
}

EnergyFunction::Values::Values(const StdVector<Residual> &residuals,
                               const Parameters &parameters,
                               const ceres::LossFunction *lossFunction,
                               PrecomputedHostToTarget &hostToTarget,
                               PrecomputedLightHostToTarget &lightHostToTarget)
    : lossFunction(lossFunction) {
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

double EnergyFunction::Values::totalEnergy(
    const StdVector<Residual> &residuals) const {
  CHECK_GT(valsAndCache.size(), 0);
  CHECK_EQ(residuals.size(), valsAndCache.size());
  int patternSize = valsAndCache[0].first.size();
  Accumulator<double> energy;
  for (int ri = 0; ri < residuals.size(); ++ri) {
    const auto &[vals, cache] = valsAndCache[ri];
    energy += residuals[ri].getDeltaEnergy(vals);
  }
  return energy.accumulated();
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

PrecomputedHostToTarget EnergyFunction::precomputeHostToTarget() const {
  return PrecomputedHostToTarget(cam, parameters.get());
}

PrecomputedMotionDerivatives
EnergyFunction::precomputeMotionDerivatives() const {
  return PrecomputedMotionDerivatives(cam, parameters.get());
}

PrecomputedLightHostToTarget
EnergyFunction::precomputeLightHostToTarget() const {
  return PrecomputedLightHostToTarget(parameters.get());
}

double
EnergyFunction::predictEnergyViaJacobian(const DeltaParameterVector &delta) {
  Values &values = computeValues();
  Accumulator<double> predictedEnergy;
  for (int ri = 0; ri < residuals.size(); ++ri) {
    VecRt predictedR =
        values.values(ri) + getPredictedResidualIncrement(ri, delta);
    predictedEnergy += residuals[ri].getDeltaEnergy(predictedR);
  }
  return predictedEnergy.accumulated();
}

EnergyFunction::Values
EnergyFunction::createValues(PrecomputedHostToTarget &hostToTarget,
                             PrecomputedLightHostToTarget &lightHostToTarget) {
  return Values(residuals, *parameters, lossFunction.get(), hostToTarget,
                lightHostToTarget);
}

EnergyFunction::Values &
EnergyFunction::computeValues(PrecomputedHostToTarget &hostToTarget,
                              PrecomputedLightHostToTarget &lightHostToTarget) {
  if (!values)
    values.emplace(residuals, *parameters, lossFunction.get(), hostToTarget,
                   lightHostToTarget);
  return values.value();
}

EnergyFunction::Values &EnergyFunction::computeValues() {
  if (!values) {
    PrecomputedHostToTarget hostToTarget = precomputeHostToTarget();
    PrecomputedLightHostToTarget lightHostToTarget =
        precomputeLightHostToTarget();
    computeValues(hostToTarget, lightHostToTarget);
  }
  return values.value();
}

EnergyFunction::Derivatives EnergyFunction::createDerivatives(
    const Values &values, PrecomputedHostToTarget &hostToTarget,
    PrecomputedMotionDerivatives &motionDerivatives,
    PrecomputedLightHostToTarget &lightHostToTarget) {
  return Derivatives(*parameters, residuals, values, hostToTarget,
                     motionDerivatives, lightHostToTarget);
}

EnergyFunction::Derivatives &EnergyFunction::computeDerivatives(
    PrecomputedHostToTarget &hostToTarget,
    PrecomputedMotionDerivatives &motionDerivatives,
    PrecomputedLightHostToTarget &lightHostToTarget) {
  if (!derivatives)
    derivatives.emplace(*parameters, residuals,
                        computeValues(hostToTarget, lightHostToTarget),
                        hostToTarget, motionDerivatives, lightHostToTarget);
  return derivatives.value();
}

EnergyFunction::Derivatives &EnergyFunction::computeDerivatives() {
  if (!derivatives) {
    PrecomputedHostToTarget hostToTarget = precomputeHostToTarget();
    PrecomputedLightHostToTarget lightHostToTarget =
        precomputeLightHostToTarget();
    PrecomputedMotionDerivatives motionDerivatives =
        precomputeMotionDerivatives();
    computeDerivatives(hostToTarget, motionDerivatives, lightHostToTarget);
  }
  return derivatives.value();
}

double EnergyFunction::totalEnergy() {
  return computeValues().totalEnergy(residuals);
}

T EnergyFunction::getPredictedDeltaEnergy(const Hessian &hessian,
                                          const Gradient &gradient,
                                          const DeltaParameterVector &delta) {
  return hessian.applyQuadraticForm(delta) + 2 * gradient.dot(delta);
}

std::shared_ptr<Parameters> EnergyFunction::getParameters() {
  return parameters;
}

} // namespace mdso::optimize