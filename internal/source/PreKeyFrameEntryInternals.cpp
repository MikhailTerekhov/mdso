#include "PreKeyFrameEntryInternals.h"

namespace mdso {

PreKeyFrameEntryInternals::PreKeyFrameEntryInternals(
    const mdso::ImagePyramid &pyramid,
    const mdso::Settings::Pyramid &pyrSettings) {
  for (int lvl = 0; lvl < pyrSettings.levelNum(); ++lvl) {
    Grid_t *newGrid = new (&gridsData[lvl * sizeof(Grid_t)])
        Grid_t(pyramid[lvl].data, 0, pyramid[lvl].rows, 0, pyramid[lvl].cols);
    new (&interpolatorsData[lvl * sizeof(Interpolator_t)])
        Interpolator_t(*newGrid);
  }
}

PreKeyFrameEntryInternals::Grid_t &PreKeyFrameEntryInternals::grid(int lvl) {
  CHECK(lvl >= 0 && lvl < pyrSettings.levelNum());
  return *reinterpret_cast<Grid_t *>(&gridsData[lvl * sizeof(Grid_t)]);
}

const PreKeyFrameEntryInternals::Grid_t &
PreKeyFrameEntryInternals::grid(int lvl) const {
  CHECK(lvl >= 0 && lvl < pyrSettings.levelNum());
  return *reinterpret_cast<const Grid_t *>(&gridsData[lvl * sizeof(Grid_t)]);
}

PreKeyFrameEntryInternals::Interpolator_t &
PreKeyFrameEntryInternals::interpolator(int lvl) {
  CHECK(lvl >= 0 && lvl < pyrSettings.levelNum());
  return *reinterpret_cast<Interpolator_t *>(
      &interpolatorsData[lvl * sizeof(Interpolator_t)]);
}

const PreKeyFrameEntryInternals::Interpolator_t &
PreKeyFrameEntryInternals::interpolator(int lvl) const {
  CHECK(lvl >= 0 && lvl < pyrSettings.levelNum());
  return *reinterpret_cast<const Interpolator_t *>(
      &interpolatorsData[lvl * sizeof(Interpolator_t)]);
}

} // namespace mdso
