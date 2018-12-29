#include "util/ImagePyramid.h"
#include "util/util.h"

namespace fishdso {

std::array<cv::Mat1b, settingPyrLevels>
createPyramid(const cv::Mat1b &baseImage) {
  std::array<cv::Mat1b, settingPyrLevels> images;
  return images;
}

ImagePyramid::ImagePyramid(const cv::Mat1b &baseImage) {
  images[0] = baseImage;
  for (int lvl = 1; lvl < settingPyrLevels; ++lvl)
    images[lvl] = boxFilterPyrUp<unsigned char>(images[lvl - 1]);
  for (int lvl = 0; lvl < settingPyrLevels; ++lvl)
    grids[lvl] = std::make_unique<ceres::Grid2D<unsigned char, 1>>(
        images[lvl].data, 0, images[lvl].rows, 0, images[lvl].cols);
  for (int lvl = 0; lvl < settingPyrLevels; ++lvl)
    interpolators[lvl] = std::make_unique<
        ceres::BiCubicInterpolator<ceres::Grid2D<unsigned char, 1>>>(
        *grids[lvl]);
}

} // namespace fishdso
