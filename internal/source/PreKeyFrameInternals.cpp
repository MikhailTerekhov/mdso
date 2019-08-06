#include "PreKeyFrameInternals.h"

namespace fishdso {

PreKeyFrameInternals::PreKeyFrameInternals(
    const ImagePyramid *pyrRefs[], int size,
    const Settings::Pyramid &pyrSettings) {
  CHECK(size > 0 && size <= Settings::CameraBundle::max_camerasInBundle);

  for (int i = 0; i < size; ++i)
    frames.emplace_back(*pyrRefs[i], pyrSettings);
}

PreKeyFrameInternals::FrameEntry::FrameEntry(
    const ImagePyramid &pyramid, const Settings::Pyramid &pyrSettings)
    : pyrSettings(pyrSettings) {
  for (int lvl = 0; lvl < pyrSettings.levelNum(); ++lvl) {
    Grid_t *newGrid = new (&gridsData[lvl * sizeof(Grid_t)])
        Grid_t(pyramid[lvl].data, 0, pyramid[lvl].rows, 0, pyramid[lvl].cols);
    new (&interpolatorsData[lvl * sizeof(Interpolator_t)])
        Interpolator_t(*newGrid);
  }
}

PreKeyFrameInternals::Grid_t &PreKeyFrameInternals::FrameEntry::grid(int lvl) {
  CHECK(lvl >= 0 && lvl < pyrSettings.levelNum());
  return *reinterpret_cast<Grid_t *>(&gridsData[lvl * sizeof(Grid_t)]);
}

const PreKeyFrameInternals::Grid_t &
PreKeyFrameInternals::FrameEntry::grid(int lvl) const {
  CHECK(lvl >= 0 && lvl < pyrSettings.levelNum());
  return *reinterpret_cast<const Grid_t *>(&gridsData[lvl * sizeof(Grid_t)]);
}

PreKeyFrameInternals::Interpolator_t &
PreKeyFrameInternals::FrameEntry::interpolator(int lvl) {
  CHECK(lvl >= 0 && lvl < pyrSettings.levelNum());
  return *reinterpret_cast<Interpolator_t *>(
      &interpolatorsData[lvl * sizeof(Interpolator_t)]);
}

const PreKeyFrameInternals::Interpolator_t &
PreKeyFrameInternals::FrameEntry::interpolator(int lvl) const {
  CHECK(lvl >= 0 && lvl < pyrSettings.levelNum());
  return *reinterpret_cast<const Interpolator_t *>(
      &interpolatorsData[lvl * sizeof(Interpolator_t)]);
}

} // namespace fishdso
