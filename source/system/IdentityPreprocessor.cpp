#include "system/IdentityPreprocessor.h"

namespace mdso {

void IdentityPreprocessor::process(cv::Mat1b multiFrame[], cv::Mat1b output[],
                                   int size) const {
  for (int i = 0; i < size; ++i)
    output[i] = multiFrame[i].clone();
}

} // namespace mdso
