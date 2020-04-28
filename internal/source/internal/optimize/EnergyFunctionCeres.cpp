#include "internal/optimize/EnergyFunctionCeres.h"
#include "optimize/SphericalPlus.h"
#include "optimize/parametrizations.h"
#include "system/KeyFrame.h"
#include "system/Reprojector.h"

#define PS (settings.residualPattern.pattern().size())
#define PH (settings.residualPattern.height)

namespace mdso::optimize {

EnergyFunctionCeres::ResidualCeres::ResidualCeres(
    CameraBundle *cameraBundle, KeyFrameEntry *hostKfEntry,
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

void logOobStatistics(
    const std::vector<EnergyFunctionCeres::ResidualCeres *> &residuals,
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
    : constParameters(keyFrames[0])
    , parameters(keyFrames[0]->preKeyFrame->cam, keyFrames, numKeyFrames)
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

  int pointsTotal = 0, pointsOOB = 0;

  for (int frameInd = 0; frameInd < numKeyFrames; ++frameInd) {
    KeyFrame *keyFrame = keyFrames[frameInd];

    auto [so3Data, tData] = getFrameData(frameInd);
    mProblem.AddParameterBlock(so3Data, 4,
                               new ceres::EigenQuaternionParameterization());
    mProblem.AddParameterBlock(tData, 3);

    ordering->AddElementToGroup(so3Data, 1);
    ordering->AddElementToGroup(tData, 1);

    for (int camInd = 0; camInd < parameters.numCameras(); ++camInd) {
      double *affLight = getAffLightData(frameInd, camInd);
      mProblem.AddParameterBlock(affLight, 2);
      mProblem.SetParameterLowerBound(affLight, 0,
                                      settings.affineLight.minAffineLightA);
      mProblem.SetParameterUpperBound(affLight, 0,
                                      settings.affineLight.maxAffineLightA);
      mProblem.SetParameterLowerBound(affLight, 1,
                                      settings.affineLight.minAffineLightB);
      mProblem.SetParameterUpperBound(affLight, 1,
                                      settings.affineLight.maxAffineLightB);
      if (!settings.affineLight.optimizeAffineLight)
        mProblem.SetParameterBlockConstant(affLight);
      ordering->AddElementToGroup(affLight, 1);
    }
  }

  mProblem.SetParameterBlockConstant(constParameters.firstToWorld.so3().data());
  mProblem.SetParameterBlockConstant(
      constParameters.firstToWorld.translation().data());
  for (int camInd = 0; camInd < parameters.numCameras(); ++camInd)
    mProblem.SetParameterBlockConstant(
        constParameters.lightWorldToFirst[camInd].data);

  auto &s2 = parameters.stateRef().secondFrame.s2();
  mProblem.SetParameterization(
      s2.data(), new ceres::AutoDiffLocalParameterization<SphericalPlus, 3, 2>(
                     new SphericalPlus(s2.center(), s2.radius(), s2.value())));

  if (settings.optimization.fixedRotationOnSecondKF)
    mProblem.SetParameterBlockConstant(
        parameters.stateRef().secondFrame.so3().data());

  if (settings.optimization.fixedMotionOnFirstAdjustent && numKeyFrames == 2) {
    mProblem.SetParameterBlockConstant(
        parameters.stateRef().secondFrame.so3().data());
    mProblem.SetParameterBlockConstant(
        parameters.stateRef().secondFrame.s2().data());
  }

  int numPoints = 0;
  int numBadDepths = 0, numOobSmallDepths = 0, numOobLargeDepths = 0;
  std::vector<OptimizedPoint *> optimizedPoints;
  Array2d<std::vector<int>> globalPointInds(
      boost::extents[parameters.numKeyFrames()][parameters.numCameras()]);
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
  parameters.setPoints(optimizedPoints);

  for (int pointInd = 0; pointInd < parameters.numPoints(); ++pointInd) {
    double *depthParamPtr = &parameters.stateRef().logDepths[pointInd];
    mProblem.AddParameterBlock(depthParamPtr, 1);
    ordering->AddElementToGroup(depthParamPtr, 0);
  }

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
      CHECK_LT(globalPointInd, parameters.numPoints());
      CHECK_EQ(optimizedPoints[globalPointInd], &op);
      CHECK(targetCam.isOnImage(repr.reprojected, PH));
      CHECK(cam->bundle[hostCamInd].cam.isOnImage(op.p, PH))
          << "Optimized point is not on the image! p = " << op.p.transpose()
          << "d = " << op.depth() << " min = " << op.minDepth
          << " max = " << op.maxDepth;
      double *depthParamPtr = &parameters.stateRef().logDepths[globalPointInd];

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
        mProblem.AddResidualBlock(
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

// for (int pointInd = 0; pointInd < parameters.numPoints(); ++pointInd) {
// double *depthParam = &parameters.stateRef().logDepths[pointInd];
// mProblem.AddParameterBlock(depthParam, 1);
// ordering->AddElementToGroup(depthParam, 0);
//}

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

void EnergyFunctionCeres::applyParameterUpdate() { parameters.apply(); }

ceres::Problem &EnergyFunctionCeres::problem() { return mProblem; }

const EnergyFunctionCeres::ResidualCeres &
EnergyFunctionCeres::residual(int residualInd) const {
  CHECK_GE(residualInd, 0);
  CHECK_LT(residualInd, residualsCeres.size());
  return *residualsCeres[residualInd];
}
std::shared_ptr<ceres::ParameterBlockOrdering>
EnergyFunctionCeres::parameterBlockOrdering() const {
  return ordering;
}
EnergyFunctionCeres::MotionData
EnergyFunctionCeres::getFrameData(int frameInd) {
  CHECK_GE(frameInd, 0);
  CHECK_LT(frameInd, parameters.numKeyFrames());

  if (frameInd == 0) {
    SE3 &firstToWorld = constParameters.firstToWorld;
    return {firstToWorld.so3().data(), firstToWorld.translation().data()};
  } else if (frameInd == 1) {
    auto &parametrization = parameters.stateRef().secondFrame;
    return {parametrization.so3().data(), parametrization.s2().data()};
  } else {
    auto &parametrization =
        parameters.stateRef().frameParametrization(frameInd);
    return {parametrization.so3().data(), parametrization.t().data()};
  }
}

double *EnergyFunctionCeres::getAffLightData(int frameInd, int camInd) {
  CHECK_GE(frameInd, 0);
  CHECK_LT(frameInd, parameters.numKeyFrames());
  CHECK_GE(camInd, 0);
  CHECK_LT(camInd, parameters.numCameras());
  if (frameInd == 0)
    return constParameters.lightWorldToFirst[camInd].data;
  else
    return parameters.stateRef().lightWorldToFrame(frameInd, camInd).data;
}

EnergyFunctionCeres::ConstParameters::ConstParameters(KeyFrame *firstFrame)
    : firstToWorld(firstFrame->thisToWorld()) {
  int numCams = firstFrame->preKeyFrame->cam->bundle.size();
  lightWorldToFirst.reserve(numCams);
  for (const KeyFrameEntry &entry : firstFrame->frames)
    lightWorldToFirst.push_back(entry.lightWorldToThis);
}

} // namespace mdso::optimize