#ifndef INCLUDE_PREPROCESSOR
#define INCLUDE_PREPROCESSOR

#include <opencv2/opencv.hpp>

namespace mdso {

class Preprocessor {
public:
  virtual ~Preprocessor();

  virtual void process(cv::Mat1b multiFrame[], cv::Mat1b output[],
                       int size) = 0;
};

} // namespace mdso

#endif
