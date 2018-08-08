#include "frontend.h"

#include "../util/defs.h"
#include "../util/settings.h"
#include "../util/util.h"
#include <algorithm>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>

namespace fishdso {

void selectCandidatePointsInternal(cv::Mat const &gradNorm,
                                   const int selBlockSize,
                                   const double threshold,
                                   std::vector<cv::Point> &cands) {
  for (int i = 0; i + selBlockSize < gradNorm.rows; i += selBlockSize)
    for (int j = 0; j + selBlockSize < gradNorm.cols; j += selBlockSize) {
      cv::Mat block = gradNorm(cv::Range(i, i + selBlockSize),
                               cv::Range(j, j + selBlockSize));
      double avg = cv::sum(block)[0] / (selBlockSize * selBlockSize);
      double mx = 0;
      cv::Point maxLoc = cv::Point(0, 0);
      cv::minMaxLoc(block, NULL, &mx, NULL, &maxLoc);
      if (mx > avg + threshold)
        cands.push_back(cv::Point(j, i) + maxLoc);
    }
}

void selectCandidatePoints(cv::Mat const &gradNorm, const int selBlockSize,
                           std::vector<cv::Point> &cands1,
                           std::vector<cv::Point> &cands2,
                           std::vector<cv::Point> &cands3) {
  int secondStart = 0, thirdStart = 0;
  selectCandidatePointsInternal(gradNorm, 1 * selBlockSize,
                                settingGradThreshold1, cands1);
  secondStart = cands1.size();
  selectCandidatePointsInternal(gradNorm, 2 * selBlockSize,
                                settingGradThreshold2, cands1);
  thirdStart = cands1.size();
  selectCandidatePointsInternal(gradNorm, 4 * selBlockSize,
                                settingGradThreshold3, cands1);

  auto less = [](cv::Point a, cv::Point b) {
    return (a.x != b.x) ? (a.x < b.x) : (a.y < b.y);
  };
  std::sort(cands1.begin(), cands1.begin() + secondStart, less);
  std::sort(cands1.begin() + secondStart, cands1.begin() + thirdStart, less);
  std::sort(cands1.begin() + thirdStart, cands1.end(), less);

  cands2.resize(thirdStart - secondStart);
  cands3.resize(cands1.size() - thirdStart);
  std::vector<cv::Point>::iterator it;
  it = std::set_difference(cands1.begin() + secondStart,
                           cands1.begin() + thirdStart, cands1.begin(),
                           cands1.begin() + secondStart, cands2.begin(), less);
  cands2.resize(it - cands2.begin());
  it = std::set_difference(cands1.begin() + thirdStart, cands1.end(),
                           cands1.begin(), cands1.begin() + thirdStart,
                           cands3.begin(), less);
  cands3.resize(it - cands3.begin());
}

} // namespace fishdso
