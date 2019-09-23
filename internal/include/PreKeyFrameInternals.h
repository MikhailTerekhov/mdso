#ifndef INCLUDE_PREKEYFRAMEINTERNALS
#define INCLUDE_PREKEYFRAMEINTERNALS

#include "util/ImagePyramid.h"
#include "util/types.h"
#include <ceres/cubic_interpolation.h>

namespace mdso {

class PreKeyFrameInternals {
public:
  using Grid_t = ceres::Grid2D<unsigned char>;
  using Interpolator_t = ceres::BiCubicInterpolator<Grid_t>;

  class FrameEntry {
  public:
    FrameEntry(const ImagePyramid &imagePyramid,
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

  PreKeyFrameInternals(const ImagePyramid *pyrRefs[], int size,
                       const Settings::Pyramid &pyrSettings);

  std::vector<FrameEntry> frames;
};

} // namespace mdso

#endif
