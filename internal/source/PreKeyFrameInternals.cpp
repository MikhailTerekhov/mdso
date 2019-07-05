#include "PreKeyFrameInternals.h"

namespace fishdso {

PreKeyFrameInternals::PreKeyFrameInternals(
    const ImagePyramid &pyramid, const Settings::Pyramid &_pyrSettings)
    : pyrSettings(_pyrSettings) {
  for (int lvl = 0; lvl < pyrSettings.levelNum; ++lvl) {
    Grid_t *newGrid = new (&gridsData[lvl * sizeof(Grid_t)])
        Grid_t(pyramid[lvl].data, 0, pyramid[lvl].rows, 0, pyramid[lvl].cols);
    new (&interpolatorsData[lvl * sizeof(Interpolator_t)])
        Interpolator_t(*newGrid);
  }
}

PreKeyFrameInternals::Grid_t &PreKeyFrameInternals::grid(int lvl) {
  CHECK(lvl >= 0 && lvl < pyrSettings.levelNum);
  return *reinterpret_cast<Grid_t *>(&gridsData[lvl * sizeof(Grid_t)]);
}

const PreKeyFrameInternals::Grid_t &PreKeyFrameInternals::grid(int lvl) const {
  CHECK(lvl >= 0 && lvl < pyrSettings.levelNum);
  return *reinterpret_cast<const Grid_t *>(&gridsData[lvl * sizeof(Grid_t)]);
}

PreKeyFrameInternals::Interpolator_t &
PreKeyFrameInternals::interpolator(int lvl) {
  CHECK(lvl >= 0 && lvl < pyrSettings.levelNum);
  return *reinterpret_cast<Interpolator_t *>(
      &interpolatorsData[lvl * sizeof(Interpolator_t)]);
}

const PreKeyFrameInternals::Interpolator_t &
PreKeyFrameInternals::interpolator(int lvl) const {
  CHECK(lvl >= 0 && lvl < pyrSettings.levelNum);
  return *reinterpret_cast<const Interpolator_t *>(
      &interpolatorsData[lvl * sizeof(Interpolator_t)]);
}

} // namespace fishdso
