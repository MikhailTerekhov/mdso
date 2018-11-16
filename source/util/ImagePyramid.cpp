#include "util/ImagePyramid.h"
#include "util/util.h"

namespace fishdso {

ImagePyramid::ImagePyramid(const cv::Mat1b &baseImage) {
  images[0] = baseImage;
  for (int lvl = 1; lvl < settingPyrLevels; ++lvl)
    images[lvl] = boxFilterPyrUp<unsigned char>(images[lvl - 1]);
}

}
