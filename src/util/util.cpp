#include "util.h"
#include "settings.h"
#include <Eigen/Eigen>
#include <RelativePoseEstimator.h>
#include <algorithm>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include <sophus/sim3.hpp>
#include <utility>

namespace fishdso {

cv::Mat dbg;

void putDot(cv::Mat &img, cv::Point const &pos, cv::Scalar const &col) {
  cv::circle(img, pos, 4, col, cv::FILLED);
}

void grad(cv::Mat const &img, cv::Mat &gradX, cv::Mat &gradY,
          cv::Mat &gradNorm) {
  static float filter[] = {-1.0, 0.0, 1.0};
  static cv::Mat gradXKer(1, 3, CV_32FC1, filter);
  static cv::Mat gradYKer(3, 1, CV_32FC1, filter);

  cv::filter2D(img, gradX, CV_32F, gradXKer, cv::Point(-1, -1), 0,
               cv::BORDER_REPLICATE);
  cv::filter2D(img, gradY, CV_32F, gradYKer, cv::Point(-1, -1), 0,
               cv::BORDER_REPLICATE);
  cv::magnitude(gradX, gradY, gradNorm);
}

} // namespace fishdso
