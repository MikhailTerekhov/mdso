#ifndef INCLUDE_DOGPREPROCESSOR
#define INCLUDE_DOGPREPROCESSOR

#include "system/Preprocessor.h"

namespace mdso {

class DoGPreprocessor : public Preprocessor {
public:
  DoGPreprocessor(double sigma1, double sigma2, double multiplier);

  void process(cv::Mat1b multiFrame[], cv::Mat1b output[],
               int size) const override;

private:
  double sigma1, sigma2;
  double multiplier;
};

} // namespace mdso

#endif
