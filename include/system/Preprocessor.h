#ifndef INCLUDE_PREPROCESSOR
#define INCLUDE_PREPROCESSOR

#include <opencv2/opencv.hpp>

namespace fishdso {

class Preprocessor {
public:
  virtual void process(cv::Mat1b multiFrame[], cv::Mat1b output[],
                       int size) = 0;
};

} // namespace fishdso

#endif
