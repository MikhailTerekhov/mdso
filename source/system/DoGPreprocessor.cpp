#include "system/DoGPreprocessor.h"

namespace mdso {

DoGPreprocessor::DoGPreprocessor(double sigma1, double sigma2,
                                 double multiplier)
    : sigma1(sigma1)
    , sigma2(sigma2)
    , multiplier(multiplier) {}

void DoGPreprocessor::process(cv::Mat1b multiFrame[], cv::Mat1b output[],
                              int size) {
  for (int i = 0; i < size; ++i)
    if (sigma1 < 0.1) {
      cv::Mat blurred;
      cv::GaussianBlur(multiFrame[i], blurred, cv::Size(0, 0), 2);
      cv::addWeighted(multiFrame[i], multiplier, blurred, -multiplier, 0,
                      output[i]);
    } else {
      cv::Mat blurred1, blurred2;
      cv::GaussianBlur(multiFrame[i], blurred1, cv::Size(0, 0), sigma1);
      cv::GaussianBlur(multiFrame[i], blurred2, cv::Size(0, 0), sigma2);
      cv::addWeighted(blurred1, multiplier, blurred2, -multiplier, 0,
                      output[i]);
    }
}

} // namespace mdso
