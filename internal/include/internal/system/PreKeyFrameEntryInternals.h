#ifndef INCLUDE_PREKEYFRAMEENTRYINTERNALS
#define INCLUDE_PREKEYFRAMEENTRYINTERNALS

#include "util/BilinearInterpolator.h"
#include "util/ImagePyramid.h"
#include "util/types.h"
#include <ceres/cubic_interpolation.h>

namespace mdso {

class PreKeyFrameEntryInternals {
public:
  using Grid_t = ceres::Grid2D<unsigned char>;

#ifdef BICUBIC_INTERPOLATOR
  using Interpolator_t = ceres::BiCubicInterpolator<Grid_t>;
#else
  using Interpolator_t = BilinearInterpolator<Grid_t>;
#endif

  PreKeyFrameEntryInternals(const ImagePyramid &pyramid,
                            const Settings::Pyramid &pyrSettings);
  Grid_t &grid(int lvl);
  const Grid_t &grid(int lvl) const;
  Interpolator_t &interpolator(int lvl);
  const Interpolator_t &interpolator(int lvl) const;

private:
  alignas(alignof(Grid_t))
      uint8_t gridsData[Settings::Pyramid::max_levelNum * sizeof(Grid_t)];
  alignas(alignof(Interpolator_t)) uint8_t
      interpolatorsData[Settings::Pyramid::max_levelNum *
                        sizeof(Interpolator_t)];
  Settings::Pyramid pyrSettings;
};

} // namespace mdso

#endif