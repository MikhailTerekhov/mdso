#include "optimize/EnergyFunction.h"
#include "optimize/Accumulator.h"
#include "optimize/StepController.h"

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
    : parameters(camBundle, keyFrames, numKeyFrames)
    , lossFunction(getLoss(settings.optimization.lossType,
                           settings.residual.intensity.outlierDiff))
    , cam(camBundle)
    , settings(settings) {
  CHECK(numKeyFrames >= 2);

  int PH = settings.residual.residualPattern.height;

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
                                   lossFunction.get(), settings.residual);
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
                     curDerivatives.parametrizationJacobians.dSecondFrame *
                     delta.sndBlock();
    else
      motionDeltaR =
          dr_dqt *
          curDerivatives.parametrizationJacobians.dRestFrames[frameInd - 2] *
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
  VecRt pointDeltaR = jacobian.dr_dlogd() * delta.pointBlock(pi);
  return hostDeltaR + targetDeltaR + pointDeltaR;
}

Hessian EnergyFunction::getHessian() {
  PrecomputedHostToTarget hostToTarget(cam, &parameters);
  PrecomputedMotionDerivatives motionDerivatives(cam, &parameters);
  PrecomputedLightHostToTarget lightHostToTarget(&parameters);
  const Values &values = computeValues(hostToTarget, lightHostToTarget);
  const Derivatives &derivatives =
      computeDerivatives(hostToTarget, motionDerivatives, lightHostToTarget);
  return getHessian(values, derivatives);
}

Hessian EnergyFunction::getHessian(const Values &precomputedValues,
                                   const Derivatives &precomputedDerivatives) {
  Hessian::AccumulatedBlocks accumulatedBlocks(parameters.numKeyFrames(),
                                               parameters.camBundleSize(),
                                               parameters.numPoints());
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
  PrecomputedHostToTarget hostToTarget(cam, &parameters);
  PrecomputedMotionDerivatives motionDerivatives(cam, &parameters);
  PrecomputedLightHostToTarget lightHostToTarget(&parameters);
  const Values &valuesRef = computeValues(hostToTarget, lightHostToTarget);
  const Derivatives &derivativesRef =
      computeDerivatives(hostToTarget, motionDerivatives, lightHostToTarget);
  return getGradient(valuesRef, derivativesRef);
}

Gradient
EnergyFunction::getGradient(const Values &precomputedValues,
                            const Derivatives &precomputedDerivatives) {
  Gradient::AccumulatedBlocks accumulatedBlocks(parameters.numKeyFrames(),
                                                parameters.camBundleSize(),
                                                parameters.numPoints());

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
  PrecomputedHostToTarget hostToTarget(cam, &parameters);
  PrecomputedMotionDerivatives motionDerivatives(cam, &parameters);
  PrecomputedLightHostToTarget lightHostToTarget(&parameters);
  computeValues(hostToTarget, lightHostToTarget);
  computeDerivatives(hostToTarget, motionDerivatives, lightHostToTarget);
}

void EnergyFunction::clearPrecomputations() {
  values.reset();
  derivatives.reset();
}

Parameters::State EnergyFunction::saveState() const {
  return parameters.saveState();
}

void EnergyFunction::recoverState(const Parameters::State &oldState) {
  parameters.recoverState(oldState);
  clearPrecomputations();
}

void EnergyFunction::optimize(int maxIterations) {
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
  for (int it = 0; it < maxIterations; ++it) {
    LOG(INFO) << "it = " << it << "\n";
    TimePoint start, end;
    start = now();

    double curEnergy = curValues.totalEnergy();
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

    DeltaParameterVector delta = dampedHessian.solve(gradient);

    double predictedEnergy = predictEnergy(delta);

    Parameters::State savedState = parameters.saveState();
    parameters.update(delta);

    auto newHostToTarget = precomputeHostToTarget();
    auto newLightHostToTarget = precomputeLightHostToTarget();
    Values newValues = createValues(newHostToTarget, newLightHostToTarget);

    double newEnergy = newValues.totalEnergy();

    LOG(INFO) << "optimization step #" << it << ": curEnergy = " << curEnergy
              << " predictedEnergy = " << predictedEnergy
              << " newEnergy = " << newEnergy
              << " delta = " << newEnergy - curEnergy;

    parametersUpdated =
        stepController.newStep(curEnergy, newEnergy, predictedEnergy);
    if (parametersUpdated) {
      curValues = std::move(newValues);
      hostToTarget = std::move(newHostToTarget);
      lightHostToTarget = std::move(newLightHostToTarget);
    } else {
      parameters.recoverState(std::move(savedState));
    }

    end = now();
    LOG(INFO) << "step took " << secondsBetween(start, end);
  }

  parameters.apply();
}

double deltaEnergy(const ceres::LossFunction *lossFunciton,
                   const VecRt &residualValues) {
  double result = 0;
  for (int i = 0; i < residualValues.size(); ++i) {
    double v = residualValues[i];
    double v2 = v * v;
    double rho[3];
    lossFunciton->Evaluate(v2, rho);
    result += rho[0];
  }
  return result;
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

double EnergyFunction::Values::totalEnergy() const {
  CHECK_GT(valsAndCache.size(), 0);
  int patternSize = valsAndCache[0].first.size();
  Accumulator<double> energy;
  for (const auto &[vals, cache] : valsAndCache)
    energy += deltaEnergy(lossFunction, vals);
  return energy.accumulated();
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
                                      [parameters->camBundleSize()]) {
  //  for (int hostInd = 0; hostInd < parameters->numKeyFrames(); ++hostInd)
  //    for (int hostCa)
}

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

EnergyFunction::PrecomputedHostToTarget
EnergyFunction::precomputeHostToTarget() const {
  return PrecomputedHostToTarget(cam, &parameters);
}
EnergyFunction::PrecomputedMotionDerivatives
EnergyFunction::precomputeMotionDerivatives() const {
  return PrecomputedMotionDerivatives(cam, &parameters);
}

EnergyFunction::PrecomputedLightHostToTarget
EnergyFunction::precomputeLightHostToTarget() const {
  return PrecomputedLightHostToTarget(&parameters);
}

double EnergyFunction::predictEnergy(const DeltaParameterVector &delta) {
  Values &values = computeValues();
  Accumulator<double> predictedEnergy;
  for (int ri = 0; ri < residuals.size(); ++ri) {
    VecRt predictedR =
        values.values(ri) + getPredictedResidualIncrement(ri, delta);
    predictedEnergy += deltaEnergy(lossFunction.get(), predictedR);
  }
  return predictedEnergy.accumulated();
}

EnergyFunction::Values
EnergyFunction::createValues(PrecomputedHostToTarget &hostToTarget,
                             PrecomputedLightHostToTarget &lightHostToTarget) {
  return Values(residuals, parameters, lossFunction.get(), hostToTarget,
                lightHostToTarget);
}

EnergyFunction::Values &
EnergyFunction::computeValues(PrecomputedHostToTarget &hostToTarget,
                              PrecomputedLightHostToTarget &lightHostToTarget) {
  if (!values)
    values.emplace(residuals, parameters, lossFunction.get(), hostToTarget,
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
  return Derivatives(parameters, residuals, values, hostToTarget,
                     motionDerivatives, lightHostToTarget);
}

EnergyFunction::Derivatives &EnergyFunction::computeDerivatives(
    PrecomputedHostToTarget &hostToTarget,
    PrecomputedMotionDerivatives &motionDerivatives,
    PrecomputedLightHostToTarget &lightHostToTarget) {
  if (!derivatives)
    derivatives.emplace(parameters, residuals,
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

} // namespace mdso::optimize