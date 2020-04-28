#ifndef INCLUDE_ENERGYFUNCTIONCERES
#define INCLUDE_ENERGYFUNCTIONCERES

#include "internal/system/PreKeyFrameEntryInternals.h"
#include "optimize/EnergyFunction.h"
#include "optimize/Parameters.h"
#include "optimize/Residual.h"
#include "system/CameraBundle.h"
#include "system/KeyFrame.h"
#include <ceres/ceres.h>
#include <memory>

namespace mdso {

namespace optimize {

class Precomputations;

class EnergyFunctionCeres {
public:
  EnergyFunctionCeres(KeyFrame *newKeyFrames[], int numKeyFrames,
                      const BundleAdjusterSettings &newSettings);
  EnergyFunctionCeres(EnergyFunction &energyFunction,
                      const BundleAdjusterSettings &newSettings);
  ~EnergyFunctionCeres();

  void optimize();
  void applyParameterUpdate();

  ceres::Problem &problem();
  std::shared_ptr<ceres::ParameterBlockOrdering> parameterBlockOrdering() const;

private:
  struct ConstParameters {
    ConstParameters(KeyFrame *firstFrame);

    SE3 firstToWorld;
    std::vector<AffLight> lightWorldToFirst;
  };

  struct MotionData {
    double *so3Data;
    double *tData;
  };

  MotionData getFrameData(int frameInd);
  double *getAffLightData(int frameInd, int camInd);

  void fillProblemFrameParameters(KeyFrame *keyFrames[], int numKeyFrames);

  std::shared_ptr<Parameters> parameters;
  ConstParameters constParameters;
  StdVector<Residual> residuals;
  std::unique_ptr<Precomputations> precomputations;
  std::unique_ptr<ceres::Problem> mProblem;
  std::shared_ptr<ceres::ParameterBlockOrdering> ordering;
  BundleAdjusterSettings settings;
};

} // namespace optimize

} // namespace mdso

#endif
