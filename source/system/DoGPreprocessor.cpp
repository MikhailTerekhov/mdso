#include "system/DoGPreprocessor.h"

namespace mdso {

DoGPreprocessor::DoGPreprocessor(double sigma1, double sigma2,
                                 double multiplier)
    : sigma1(sigma1)
    , sigma2(sigma2)
    , multiplier(multiplier) {}

void DoGPreprocessor::process(cv::Mat1b multiFrame[], cv::Mat1b output[],
                              int size) {
  for (int i = 0; i < size; ++i) {
    output[i].create(multiFrame[i].size());
    if (sigma1 < 0.1) {
      cv::Mat1b blurred;
      cv::GaussianBlur(multiFrame[i], blurred, cv::Size(0, 0), 2);
      for (int y = 0; y < blurred.rows; ++y)
        for (int x = 0; x < blurred.cols; ++x)
          output[i](y, x) = 128u + multiFrame[i](y, x) - blurred(y, x);

      //      cv::addWeighted(multiFrame[i], multiplier, blurred, -multiplier, 0,
      //                      output[i]);
    } else {
      cv::Mat1b blurred1, blurred2;
      cv::GaussianBlur(multiFrame[i], blurred1, cv::Size(0, 0), sigma1);
      cv::GaussianBlur(multiFrame[i], blurred2, cv::Size(0, 0), sigma2);
      for (int y = 0; y < blurred1.rows; ++y)
        for (int x = 0; x < blurred1.cols; ++x)
          output[i](y, x) = 128u + blurred1(y, x) - blurred2(y, x);
      //      cv::addWeighted(blurred1, multiplier, blurred2, -multiplier, uint8_t(128),
      //                      output[i]);
    }
  }
}

} // namespace mdso
