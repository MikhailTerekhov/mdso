#pragma once

#include <opencv2/core.hpp>
#include <vector>

namespace fishdso {

void selectCandidatePoints(cv::Mat const &gradNorm, const int selBlockSize,
                           std::vector<cv::Point> &cands1,
                           std::vector<cv::Point> &cands2,
                           std::vector<cv::Point> &cands3);
}
