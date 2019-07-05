#include "util/ImagePyramid.h"
#include "util/util.h"

namespace fishdso {

ImagePyramid::ImagePyramid(const cv::Mat1b &baseImage, int levelNum)
    : images(levelNum) {
  images[0] = baseImage;
  for (int lvl = 1; lvl < levelNum; ++lvl)
    images[lvl] = boxFilterPyrDown<unsigned char>(images[lvl - 1]);
}

} // namespace fishdso
