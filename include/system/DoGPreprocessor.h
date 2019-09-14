#ifndef INCLUDE_DOGPREPROCESSOR
#define INCLUDE_DOGPREPROCESSOR

#include "system/Preprocessor.h"

namespace fishdso {

class DoGPreprocessor : public Preprocessor {
public:
  DoGPreprocessor(double sigma1, double sigma2, double multiplier);

  void process(cv::Mat1b multiFrame[], cv::Mat1b output[], int size);

private:
  double sigma1, sigma2;
  double multiplier;
};

} // namespace fishdso

#endif
