#include "internal/optimize/EnergyFunctionCeres.h"
#include "optimize/SphericalPlus.h"
#include "optimize/parametrizations.h"
#include "system/KeyFrame.h"
#include "system/Reprojector.h"

#define PS (settings.residualPattern.pattern().size())
#define PH (settings.residualPattern.height)

namespace mdso::optimize {

using VecR = Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::ColMajor,
                           Settings::ResidualPattern::max_size>;
using MatR2RM = Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor,
                              Settings::ResidualPattern::max_size>;
using MatR3RM = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor,
                              Settings::ResidualPattern::max_size>;
using MatR4RM = Eigen::Matrix<double, Eigen::Dynamic, 4, Eigen::RowMajor,
                              Settings::ResidualPattern::max_size>;

struct ResidualCeres {
  ResidualCeres(CameraBundle *cameraBundle, KeyFrameEntry *hostKfEntry,
                KeyFrameEntry *targetKfEntry, OptimizedPoint *optimizedPoint,
                int numInPattern, const BundleAdjusterSettings &settings)
      : cameraBundle(cameraBundle)
      , hostCam(&cameraBundle->bundle[hostKfEntry->ind].cam)
      , targetCam(&cameraBundle->bundle[targetKfEntry->ind].cam)
      , hostKf(hostKfEntry->host)
      , targetKf(targetKfEntry->host)
      , hostImage(&hostKf->preKeyFrame->frames[hostKfEntry->ind]
                       .internals->interpolator(0))
      , targetImage(&targetKf->preKeyFrame->frames[targetKfEntry->ind]
                         .internals->interpolator(0))
      , hostFrameToBody(cameraBundle->bundle[hostKfEntry->ind].thisToBody)
      , targetBodyToFrame(cameraBundle->bundle[targetKfEntry->ind].bodyToThis)
      , optimizedPoint(optimizedPoint)
      , hostDirection(optimizedPoint->dir)
      , numInPattern(numInPattern)
      , settings(settings) {
    Vec2 pixelPos =
        optimizedPoint->p + settings.residualPattern.pattern()[numInPattern];
    hostImage->Evaluate(pixelPos[1], pixelPos[0], &hostIntensity);
    Vec3 hostPatternDir = hostCam->unmap(pixelPos).normalized();
    SE3 hostToWorld = hostKf->thisToWorld();
    SE3 targetToWorld = targetKf->thisToWorld();
    double depth = optimizedPoint->depth();
    SE3 hostFrameToTargetFrame = targetBodyToFrame * targetToWorld.inverse() *
                                 hostToWorld * hostFrameToBody;
    Vec3 targetPatternDir, targetPointDir;
    if (depth > settings.depth.max) {
      targetPatternDir = hostFrameToTargetFrame.so3() * hostPatternDir;
      targetPointDir = hostFrameToTargetFrame.so3() * hostDirection;
    } else {
      targetPatternDir = hostFrameToTargetFrame * (depth * hostPatternDir);
      targetPointDir = hostFrameToTargetFrame * (depth * hostDirection);
    }
    targetPatternDelta =
        targetCam->map(targetPatternDir) - targetCam->map(targetPointDir);
  }

  template <typename T>
  bool operator()(const T *const depthParamP, const T *const hostTransP,
                  const T *const hostRotP, const T *const targetTransP,
                  const T *const targetRotP, const T *const hostAffP,
                  const T *const targetAffP, T *res) const {
    using Vec2t = Eigen::Matrix<T, 2, 1>;
    using Vec3t = Eigen::Matrix<T, 3, 1>;
    using Mat33t = Eigen::Matrix<T, 3, 3>;
    using Quatt = Eigen::Quaternion<T>;
    using SE3t = Sophus::SE3<T>;

    Eigen::Map<const Vec3t> hostTransM(hostTransP);
    Vec3t hostTrans(hostTransM);
    Eigen::Map<const Quatt> hostRotM(hostRotP);
    Quatt hostRot(hostRotM);
    SE3t hostToWorld(hostRot, hostTrans);

    Eigen::Map<const Vec3t> targetTransM(targetTransP);
    Vec3t targetTrans(targetTransM);
    Eigen::Map<const Quatt> targetRotM(targetRotP);
    Quatt targetRot(targetRotM);
    SE3t targetToWorld(targetRot, targetTrans);

    SE3t targetBodyToFrameT = targetBodyToFrame.template cast<T>();
    SE3t hostFrameToBodyT = hostFrameToBody.template cast<T>();

    const T *hostAffLightP = hostAffP;
    AffineLightTransform<T> lightWorldToHost(hostAffLightP[0],
                                             hostAffLightP[1]);

    const T *targetAffLightP = targetAffP;
    AffineLightTransform<T> lightWorldToTarget(targetAffLightP[0],
                                               targetAffLightP[1]);

    T depth = ceres::exp(*depthParamP);

    Vec3t targetPos;
    if (depth > T(settings.depth.max))
      targetPos = targetBodyToFrameT.so3() *
                  (targetToWorld.so3().inverse() *
                   (hostToWorld.so3() *
                    (hostFrameToBodyT.so3() * (hostDirection.cast<T>()))));
    else
      targetPos = targetBodyToFrameT *
                  (targetToWorld.inverse() *
                   (hostToWorld *
                    (hostFrameToBodyT * (hostDirection.cast<T>() * depth))));
    Vec2t targetPosMapped =
        targetCam->map(targetPos) + targetPatternDelta.cast<T>();
    T trackedIntensity;
    targetImage->Evaluate(targetPosMapped[1], targetPosMapped[0],
                          &trackedIntensity);
    T transformedHostIntensity =
        lightWorldToTarget(lightWorldToHost.inverse()(T(hostIntensity)));
    *res = trackedIntensity - transformedHostIntensity;

    return true;
  }

  CameraBundle *cameraBundle;
  CameraModel *hostCam;
  CameraModel *targetCam;
  KeyFrame *hostKf;
  KeyFrame *targetKf;
  PreKeyFrameEntryInternals::Interpolator_t *hostImage;
  PreKeyFrameEntryInternals::Interpolator_t *targetImage;
  SE3 hostFrameToBody, targetBodyToFrame;
  OptimizedPoint *optimizedPoint;
  Vec3 hostDirection;
  int numInPattern;
  double hostIntensity;
  Vec2 targetPatternDelta;
  const BundleAdjusterSettings &settings;
};

void logOobStatistics(const std::vector<ResidualCeres *> &residuals,
                      const BundleAdjusterSettings &settings) {
  int numOob = 0, numOobBigDepth = 0;
  for (const auto *res : residuals) {
    SE3 hostToWorld = res->hostKf->thisToWorld();
    SE3 hostCamToBody = res->hostFrameToBody;
    SE3 targetToWorld = res->targetKf->thisToWorld();
    SE3 targetBodyToCam = res->targetBodyToFrame;
    SE3 hostToTarget =
        targetBodyToCam * targetToWorld.inverse() * hostToWorld * hostCamToBody;
    double depth = res->optimizedPoint->depth();
    Vec2 reproj =
        res->targetCam->map(hostToTarget * (depth * res->optimizedPoint->dir));

    if (!res->targetCam->isOnImage(reproj, PH)) {
      numOob++;
      if (depth > settings.depth.max)
        numOobBigDepth++;
    }
  }
  LOG(INFO) << "total resduals; total OOB; OOB with big depths:";
  LOG(INFO) << residuals.size() << ' ' << numOob << ' ' << numOobBigDepth;
}

EnergyFunctionCeres::EnergyFunctionCeres(
    KeyFrame *keyFrames[], int numKeyFrames,
    const BundleAdjusterSettings &newSettings)
    : parameters(new Parameters(keyFrames[0]->preKeyFrame->cam, keyFrames,
                                numKeyFrames))
    , constParameters(keyFrames[0])
    , mProblem(new ceres::Problem())
    , ordering(new ceres::ParameterBlockOrdering())
    , settings(newSettings) {
  static_assert(
      std::is_same_v<T, double>,
      "Ceres only works with double precision. Please, configure without "
      "-DFLOAT_OPTIMIZATION");
  static_assert(std::is_same_v<FrameParametrization, SO3xR3Parametrization>,
                "Please, configure with -DSO3_X_R3_PARAMETRIZATION");
  CHECK_GE(numKeyFrames, 2);

  CameraBundle *cam = keyFrames[0]->preKeyFrame->cam;

  fillProblemFrameParameters(keyFrames, numKeyFrames);

  int numPoints = 0;
  int numBadDepths = 0, numOobSmallDepths = 0, numOobLargeDepths = 0;
  std::vector<OptimizedPoint *> optimizedPoints;
  Array2d<std::vector<int>> globalPointInds(
      boost::extents[parameters->numKeyFrames()][parameters->numCameras()]);
  for (int frameInd = 0; frameInd < numKeyFrames; ++frameInd) {
    KeyFrame *keyFrame = keyFrames[frameInd];
    for (int camInd = 0; camInd < cam->bundle.size(); ++camInd) {
      KeyFrameEntry &hostEntry = keyFrame->frames[camInd];
      globalPointInds[frameInd][camInd].resize(hostEntry.optimizedPoints.size(),
                                               -1);
      for (int pointInd = 0; pointInd < hostEntry.optimizedPoints.size();
           ++pointInd) {
        OptimizedPoint &op = hostEntry.optimizedPoints[pointInd];
        numPoints++;

        if ((settings.depth.setMinBound ||
             settings.depth.useMinPlusExpParametrization) &&
            op.depth() <= settings.depth.min) {
          numOobSmallDepths++;
          continue;
        }
        if (settings.depth.setMaxBound && op.depth() > settings.depth.max) {
          numOobLargeDepths++;
          continue;
        }
        if (op.depth() < 0 || !std::isfinite(op.depth())) {
          numBadDepths++;
          continue;
        }

        globalPointInds[frameInd][camInd][pointInd] = optimizedPoints.size();
        optimizedPoints.push_back(&op);
      }
    }
  }
  parameters->setPoints(optimizedPoints);

  for (int pointInd = 0; pointInd < parameters->numPoints(); ++pointInd) {
    double *depthParamPtr = &parameters->stateRef().logDepths[pointInd];
    mProblem->AddParameterBlock(depthParamPtr, 1);
    ordering->AddElementToGroup(depthParamPtr, 0);
  }

  std::vector<ResidualCeres *> residualsCeres;
  for (int targetInd = 0; targetInd < numKeyFrames; ++targetInd) {
    KeyFrame *targetFrame = keyFrames[targetInd];
    Reprojector<OptimizedPoint> reprojector(keyFrames, numKeyFrames,
                                            keyFrames[targetInd]->thisToWorld(),
                                            settings.depth, PH);
    reprojector.setSkippedFrame(targetInd);
    StdVector<Reprojection> reprojections = reprojector.reproject();
    for (const auto &repr : reprojections) {
      int targetCamInd = repr.targetCamInd, hostInd = repr.hostInd,
          hostCamInd = repr.hostCamInd, pointInd = repr.pointInd;
      CHECK_NE(hostInd, targetInd);
      CameraModel &targetCam = cam->bundle[targetCamInd].cam;
      KeyFrame *hostFrame = keyFrames[hostInd];
      OptimizedPoint &op =
          keyFrames[hostInd]->frames[hostCamInd].optimizedPoints[pointInd];
      int globalPointInd = globalPointInds[hostInd][hostCamInd][pointInd];
      if (globalPointInd == -1)
        continue;
      CHECK_GE(globalPointInd, 0);
      CHECK_LT(globalPointInd, parameters->numPoints());
      CHECK_EQ(optimizedPoints[globalPointInd], &op);
      CHECK(targetCam.isOnImage(repr.reprojected, PH));
      CHECK(cam->bundle[hostCamInd].cam.isOnImage(op.p, PH))
          << "Optimized point is not on the image! p = " << op.p.transpose()
          << "d = " << op.depth() << " min = " << op.minDepth
          << " max = " << op.maxDepth;
      double *depthParamPtr = &parameters->stateRef().logDepths[globalPointInd];

      for (int i = 0; i < PS; ++i) {
        auto newResidual = new ResidualCeres(
            cam, &hostFrame->frames[hostCamInd],
            &targetFrame->frames[targetCamInd], &op, i, settings);

        residualsCeres.push_back(newResidual);

        const Vec2 &shiftedP = op.p + settings.residualPattern.pattern()[i];
        double gradNorm = hostFrame->preKeyFrame->frames[hostCamInd].gradNorm(
            toCvPoint(shiftedP));
        const double c = settings.residualWeighting.c;
        double weight = settings.residualWeighting.useGradientWeighting
                            ? c / std::hypot(c, gradNorm)
                            : 1;
        ceres::LossFunction *lossFunc = new ceres::ScaledLoss(
            new ceres::HuberLoss(settings.intensity.outlierDiff), weight,
            ceres::Ownership::TAKE_OWNERSHIP);

        MotionData hostBodyToWorld = getFrameData(hostInd);
        MotionData targetBodyToWorld = getFrameData(targetInd);
        mProblem->AddResidualBlock(
            new ceres::AutoDiffCostFunction<ResidualCeres, 1, 1, 3, 4, 3, 4, 2,
                                            2>(newResidual),
            lossFunc, depthParamPtr, hostBodyToWorld.tData,
            hostBodyToWorld.so3Data, targetBodyToWorld.tData,
            targetBodyToWorld.so3Data, getAffLightData(hostInd, hostCamInd),
            getAffLightData(targetInd, targetCamInd));
      }
    }
  }

  LOG(INFO) << "total points = " << numPoints
            << " OOB small = " << numOobSmallDepths
            << " OOB large = " << numOobLargeDepths
            << " bad = " << numBadDepths;

  logOobStatistics(residualsCeres, settings);
}

struct JacobianBlocks {
  JacobianBlocks(const Residual::Jacobian &jacobian)
      : dr_dq_host(jacobian.dr_dq_host())
      , dr_dt_host(jacobian.dr_dt_host())
      , dr_dq_target(jacobian.dr_dq_target())
      , dr_dt_target(jacobian.dr_dt_target())
      , dr_daff_host(jacobian.dr_daff_host())
      , dr_daff_target(jacobian.dr_daff_target()) {}

  MatR4RM dr_dq_host;
  MatR3RM dr_dt_host;
  MatR4RM dr_dq_target;
  MatR3RM dr_dt_target;
  MatR2RM dr_daff_host;
  MatR2RM dr_daff_target;
  VecR dr_dlogd;
};

class Precomputations : public ceres::EvaluationCallback {
public:
  Precomputations(EnergyFunction &energyFunction)
      : energyFunction(energyFunction) {}

  void PrepareForEvaluation(bool evaluate_jacobians,
                            bool new_evaluation_point) override {
    if (new_evaluation_point) {
      hostToTarget.emplace(energyFunction.precomputeHostToTarget());
      lightHostToTarget.emplace(energyFunction.precomputeLightHostToTarget());
      values.emplace(
          energyFunction.createValues(*hostToTarget, *lightHostToTarget));
    }
    if (evaluate_jacobians) {
      motionDerivatives.emplace(energyFunction.precomputeMotionDerivatives());
      auto jacobians =
          energyFunction
              .createDerivatives(*values, *hostToTarget, *motionDerivatives,
                                 *lightHostToTarget)
              .residualJacobians;
      derivatives.emplace();
      derivatives->reserve(jacobians.size());
      for (const Residual::Jacobian &jacobian : jacobians)
        derivatives->emplace_back(jacobian);
    }
  }

  double value(int residualInd, int patternInd) const {
    CHECK_GE(residualInd, 0);
    CHECK_LT(residualInd, derivatives->size());
    CHECK(values);
    const VecR &valuesBlock = values->values(residualInd);
    CHECK_GE(patternInd, 0);
    CHECK_LT(patternInd, valuesBlock.size());
    return valuesBlock[residualInd];
  }

  const JacobianBlocks &jacobianBlocks(int residualInd) const {
    CHECK(derivatives);
    CHECK_GE(residualInd, 0);
    CHECK_LT(residualInd, derivatives->size());
    return (*derivatives)[residualInd];
  }

private:
  EnergyFunction &energyFunction;
  int patternSize;
  std::optional<PrecomputedHostToTarget> hostToTarget;
  std::optional<PrecomputedLightHostToTarget> lightHostToTarget;
  std::optional<PrecomputedMotionDerivatives> motionDerivatives;
  std::optional<EnergyFunction::Values> values;
  std::optional<StdVector<JacobianBlocks>> derivatives;
};

class ResidualPrecomputed
    : public ceres::SizedCostFunction<1, 1, 3, 4, 3, 4, 2, 2> {
public:
  ResidualPrecomputed(Precomputations *precomputations, int residualInd,
                      int patternInd)
      : precomputations(precomputations)
      , residualInd(residualInd)
      , patternInd(patternInd) {}

  bool Evaluate(const double *const *parameters, double *residuals,
                double **jacobians) const override {
    residuals[0] = precomputations->value(residualInd, patternInd);
    if (jacobians) {
      const JacobianBlocks &jacobianBlocks =
          precomputations->jacobianBlocks(residualInd);
      if (jacobians[0])
        *jacobians[0] = jacobianBlocks.dr_dlogd[patternInd];
      if (jacobians[1]) {
        Eigen::Map<Vec3> dri_dt_host(jacobians[1]);
        dri_dt_host = jacobianBlocks.dr_dt_host.row(patternInd);
      }
      if (jacobians[2]) {
        Eigen::Map<Vec4> dri_dq_host(jacobians[2]);
        dri_dq_host = jacobianBlocks.dr_dq_host.row(patternInd);
      }
      if (jacobians[3]) {
        Eigen::Map<Vec3> dri_dt_target(jacobians[3]);
        dri_dt_target = jacobianBlocks.dr_dt_target.row(patternInd);
      }
      if (jacobians[4]) {
        Eigen::Map<Vec4> dri_dq_target(jacobians[4]);
        dri_dq_target = jacobianBlocks.dr_dq_target.row(patternInd);
      }
      if (jacobians[5]) {
        Eigen::Map<Vec2> dri_daff_host(jacobians[5]);
        dri_daff_host = jacobianBlocks.dr_daff_host.row(patternInd);
      }
      if (jacobians[6]) {
        Eigen::Map<Vec2> dri_daff_target(jacobians[6]);
        dri_daff_target = jacobianBlocks.dr_daff_target.row(patternInd);
      }
    }
    return true;
  }

private:
  Precomputations *precomputations;
  int residualInd;
  int patternInd;
};

EnergyFunctionCeres::EnergyFunctionCeres(
    EnergyFunction &energyFunction, const BundleAdjusterSettings &newSettings)
    : parameters(energyFunction.getParameters())
    , constParameters(parameters->getKeyFrames()[0])
    , residuals(energyFunction.getResiduals())
    , precomputations(new Precomputations(energyFunction))
    , ordering(new ceres::ParameterBlockOrdering())
    , settings(newSettings) {
  ceres::Problem::Options problemOptions;
  problemOptions.evaluation_callback = precomputations.get();
  mProblem.reset(new ceres::Problem(problemOptions));

  std::vector<KeyFrame *> keyFrames = parameters->getKeyFrames();
  fillProblemFrameParameters(keyFrames.data(), keyFrames.size());
  for (int pointInd = 0; pointInd < parameters->numPoints(); ++pointInd) {
    double *depthParam = &parameters->stateRef().logDepths[pointInd];
    mProblem->AddParameterBlock(depthParam, 1);
    ordering->AddElementToGroup(depthParam, 0);
  }

  CHECK_GT(residuals.size(), 0);
  int patternSize = settings.residualPattern.pattern().size();
  for (int residualInd = 0; residualInd < residuals.size(); ++residualInd) {
    const Residual &res = residuals[residualInd];
    int hostInd = res.hostInd(), hostCamInd = res.hostCamInd(),
        targetInd = res.targetInd(), targetCamInd = res.targetCamInd(),
        pointInd = res.pointInd();
    MotionData hostBodyToWorld = getFrameData(hostInd);
    MotionData targetBodyToWorld = getFrameData(targetInd);
    double *hostLight = getAffLightData(hostInd, hostCamInd);
    double *targetLight = getAffLightData(targetInd, targetCamInd);
    double *logDepth = &parameters->stateRef().logDepths[pointInd];
    VecRt gradWeights = residuals[residualInd].getPixelDependentWeights();
    for (int patternInd = 0; patternInd < patternSize; ++patternInd) {
      ceres::LossFunction *lossFunc = new ceres::ScaledLoss(
          new ceres::HuberLoss(settings.intensity.outlierDiff),
          gradWeights[patternInd], ceres::Ownership::TAKE_OWNERSHIP);
      mProblem->AddResidualBlock(
          new ResidualPrecomputed(precomputations.get(), residualInd,
                                  patternInd),
          lossFunc, logDepth, hostBodyToWorld.tData, hostBodyToWorld.so3Data,
          targetBodyToWorld.tData, targetBodyToWorld.so3Data, hostLight,
          targetLight);
    }
  }
}

void EnergyFunctionCeres::fillProblemFrameParameters(KeyFrame *keyFrames[],
                                                     int numKeyFrames) {
  for (int frameInd = 0; frameInd < numKeyFrames; ++frameInd) {
    KeyFrame *keyFrame = keyFrames[frameInd];

    auto [so3Data, tData] = getFrameData(frameInd);
    mProblem->AddParameterBlock(so3Data, 4,
                                new ceres::EigenQuaternionParameterization());
    mProblem->AddParameterBlock(tData, 3);

    ordering->AddElementToGroup(so3Data, 1);
    ordering->AddElementToGroup(tData, 1);

    for (int camInd = 0; camInd < parameters->numCameras(); ++camInd) {
      double *affLight = getAffLightData(frameInd, camInd);
      mProblem->AddParameterBlock(affLight, 2);
      mProblem->SetParameterLowerBound(affLight, 0,
                                       settings.affineLight.minAffineLightA);
      mProblem->SetParameterUpperBound(affLight, 0,
                                       settings.affineLight.maxAffineLightA);
      mProblem->SetParameterLowerBound(affLight, 1,
                                       settings.affineLight.minAffineLightB);
      mProblem->SetParameterUpperBound(affLight, 1,
                                       settings.affineLight.maxAffineLightB);
      if (!settings.affineLight.optimizeAffineLight)
        mProblem->SetParameterBlockConstant(affLight);
      ordering->AddElementToGroup(affLight, 1);
    }
  }

  mProblem->SetParameterBlockConstant(
      constParameters.firstToWorld.so3().data());
  mProblem->SetParameterBlockConstant(
      constParameters.firstToWorld.translation().data());
  for (int camInd = 0; camInd < parameters->numCameras(); ++camInd)
    mProblem->SetParameterBlockConstant(
        constParameters.lightWorldToFirst[camInd].data);

  auto &s2 = parameters->stateRef().secondFrame.s2();
  mProblem->SetParameterization(
      s2.data(), new ceres::AutoDiffLocalParameterization<SphericalPlus, 3, 2>(
                     new SphericalPlus(s2.center(), s2.radius(), s2.value())));

  if (settings.optimization.fixedRotationOnSecondKF)
    mProblem->SetParameterBlockConstant(
        parameters->stateRef().secondFrame.so3().data());

  if (settings.optimization.fixedMotionOnFirstAdjustent && numKeyFrames == 2) {
    mProblem->SetParameterBlockConstant(
        parameters->stateRef().secondFrame.so3().data());
    mProblem->SetParameterBlockConstant(
        parameters->stateRef().secondFrame.s2().data());
  }
}

void EnergyFunctionCeres::optimize() {
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.linear_solver_ordering = ordering;
  options.max_num_iterations = settings.optimization.maxIterations;
  options.num_threads = settings.threading.numThreads;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem(), &summary);
  LOG(INFO) << summary.FullReport() << std::endl;
}

void EnergyFunctionCeres::applyParameterUpdate() { parameters->apply(); }

ceres::Problem &EnergyFunctionCeres::problem() { return *mProblem; }

std::shared_ptr<ceres::ParameterBlockOrdering>
EnergyFunctionCeres::parameterBlockOrdering() const {
  return ordering;
}
EnergyFunctionCeres::MotionData
EnergyFunctionCeres::getFrameData(int frameInd) {
  CHECK_GE(frameInd, 0);
  CHECK_LT(frameInd, parameters->numKeyFrames());

  if (frameInd == 0) {
    SE3 &firstToWorld = constParameters.firstToWorld;
    return {firstToWorld.so3().data(), firstToWorld.translation().data()};
  } else if (frameInd == 1) {
    auto &parametrization = parameters->stateRef().secondFrame;
    return {parametrization.so3().data(), parametrization.s2().data()};
  } else {
    auto &parametrization =
        parameters->stateRef().frameParametrization(frameInd);
    return {parametrization.so3().data(), parametrization.t().data()};
  }
}

double *EnergyFunctionCeres::getAffLightData(int frameInd, int camInd) {
  CHECK_GE(frameInd, 0);
  CHECK_LT(frameInd, parameters->numKeyFrames());
  CHECK_GE(camInd, 0);
  CHECK_LT(camInd, parameters->numCameras());
  if (frameInd == 0)
    return constParameters.lightWorldToFirst[camInd].data;
  else
    return parameters->stateRef().lightWorldToFrame(frameInd, camInd).data;
}

EnergyFunctionCeres::ConstParameters::ConstParameters(KeyFrame *firstFrame)
    : firstToWorld(firstFrame->thisToWorld()) {
  int numCams = firstFrame->preKeyFrame->cam->bundle.size();
  lightWorldToFirst.reserve(numCams);
  for (const KeyFrameEntry &entry : firstFrame->frames)
    lightWorldToFirst.push_back(entry.lightWorldToThis);
}

EnergyFunctionCeres::~EnergyFunctionCeres() = default;

} // namespace mdso::optimize