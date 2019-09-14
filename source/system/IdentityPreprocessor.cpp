#include "system/IdentityPreprocessor.h"

namespace fishdso {

void IdentityPreprocessor::process(cv::Mat1b multiFrame[], cv::Mat1b output[], int size) {
  for (int i = 0; i < size; ++i)
    output[i] = multiFrame[i].clone();
}

}
