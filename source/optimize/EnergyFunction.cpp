#include "optimize/EnergyFunction.h"

#define PH (settings.residualPattern.height)

namespace mdso::optimize {

SO3xS2Parametrization getSecondFrameParam(const SE3 &f1ToWorld,
                                          const SE3 &f2ToWorld) {
  return SO3xS2Parametrization(f2ToWorld.so3().cast<T>(),
                               f1ToWorld.translation().cast<T>(),
                               f2ToWorld.translation().cast<T>());
}

EnergyFunction::EnergyFunction(CameraBundle *camBundle, KeyFrame **keyFrames,
                               int size, const ResidualSettings &settings)
    : firstFrame(keyFrames[0])
    , secondFrame(keyFrames[1])
    , secondFrameParametrization(getSecondFrameParam(
          keyFrames[0]->thisToWorld(), keyFrames[1]->thisToWorld()))
    , settings(settings) {
  CHECK(size >= 2);

  frames.reserve(size - 2);
  for (int baseInd = 2; baseInd < size; ++baseInd)
    frames.push_back({RightExpParametrization<SE3t>(
                          keyFrames[baseInd]->thisToWorld().cast<T>()),
                      keyFrames[baseInd]});

  ceres::HuberLoss loss(settings.intensity.outlierDiff);
  CameraModel *cam = &camBundle->bundle[0].cam;

  Array2d<SE3> baseToTarget(boost::extents[size][size]);
  for (int baseInd = 0; baseInd < size; ++baseInd)
    for (int targetInd = 0; targetInd < size; ++targetInd) {
      if (targetInd == baseInd)
        continue;
      baseToTarget[baseInd][targetInd] =
          keyFrames[targetInd]->thisToWorld().inverse() *
          keyFrames[baseInd]->thisToWorld();
    }

  for (int baseInd = 0; baseInd < size; ++baseInd)
    for (OptimizedPoint &op : keyFrames[baseInd]->frames[0].optimizedPoints) {
      Vec3 ray = op.depth() * op.dir;
      bool hasResiduals = false;
      for (int targetInd = 0; targetInd < size; ++targetInd) {
        if (baseInd == targetInd)
          continue;
        Vec3 rayTarget = baseToTarget[baseInd][targetInd] * ray;
        if (!cam->isMappable(rayTarget))
          continue;
        Vec2 pointTarget = cam->map(rayTarget);
        if (!cam->isOnImage(pointTarget, PH))
          continue;

        residuals.push_back(Residual(
            &camBundle->bundle[0], &camBundle->bundle[0],
            &keyFrames[baseInd]->frames[0], &keyFrames[targetInd]->frames[0],
            &op, baseToTarget[baseInd][targetInd], &loss, settings));
        hasResiduals = true;
      }

      if (hasResiduals)
        points.push_back({op.depth(), &op});
    }
}

EnergyFunction::Hessian EnergyFunction::getHessian() {
  Hessian hessian;
  int framePars = frames.size() * Hessian::framePars + Hessian::secondFramePars;
  int pointPars = points.size();
  hessian.frameFrame = MatXX(framePars, framePars);
  hessian.framePoint = MatXX(framePars, pointPars);
  hessian.pointPoint = VecX(pointPars);

  for (Residual &residual : residuals) {
    // TODO
  }

  return hessian;
}

}