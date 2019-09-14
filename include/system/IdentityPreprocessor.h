#ifndef INCLUDE_IDENTITYPREPROCESSOR
#define INCLUDE_IDENTITYPREPROCESSOR

#include "system/Preprocessor.h"

namespace fishdso {

class IdentityPreprocessor : public Preprocessor {
public:
  void process(cv::Mat1b multiFrame[], cv::Mat1b output[], int size);
};

} // namespace fishdso

#endif
