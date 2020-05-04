#ifndef INCLUDE_ENERGYFUNCTIONCERES
#define INCLUDE_ENERGYFUNCTIONCERES

#include "internal/system/PreKeyFrameEntryInternals.h"
#include "system/CameraBundle.h"
#include "system/KeyFrame.h"
#include <ceres/ceres.h>
#include <memory>

namespace mdso {

namespace optimize {

class EnergyFunctionCeres {
public:
  struct Residual {
    Residual(CameraBundle *cameraBundle, KeyFrameEntry *hostKfEntry,
             KeyFrameEntry *targetKfEntry, OptimizedPoint *optimizedPoint,
             int numInPattern, const BundleAdjusterSettings &settings);

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

      T depth = settings.depth.useMinPlusExpParametrization
                    ? settings.depth.min + ceres::exp(*depthParamP)
                    : ceres::exp(*depthParamP);

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

  EnergyFunctionCeres(KeyFrame *newKeyFrames[], int numKeyFrames,
                      const BundleAdjusterSettings &newSettings);

  void optimize();
  void applyParameterUpdate();

  ceres::Problem &problem();
  const Residual &residual(int residualInd) const;
  std::shared_ptr<ceres::ParameterBlockOrdering> parameterBlockOrdering() const;

private:
  struct PointParam {
    double depthParam;
    OptimizedPoint *op;
  };

  StdVector<SE3> bodyToWorld;
  std::vector<PointParam> pointParams;

  std::vector<KeyFrame *> keyFrames;
  std::vector<Residual *> residuals;
  ceres::Problem mProblem;
  std::shared_ptr<ceres::ParameterBlockOrdering> ordering;
  BundleAdjusterSettings settings;
};

} // namespace optimize

} // namespace mdso

#endif
