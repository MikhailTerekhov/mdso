#include "system/keyframe.h"
#include "frontend/frontend.h"
#include "system/interestpoint.h"
#include "util/defs.h"
#include "util/settings.h"
#include "util/util.h"
#include <opencv2/opencv.hpp>

namespace fishdso {

KeyFrame::KeyFrame(int frameId, cv::Mat frameColored, DsoSystem *dsoSystem)
    : frameId(frameId), frameColored(frameColored), dsoSystem(dsoSystem) {

  cv::cvtColor(frameColored, frame, cv::COLOR_BGR2GRAY);
  grad(frame, gradX, gradY, gradNorm);

  selectPoints();
}

void KeyFrame::selectPoints() {
  std::vector<cv::Point> cands1, cands2, cands3;
  selectCandidatePoints(gradNorm, dsoSystem->adaptiveBlockSize, cands1, cands2,
                        cands3);
  dsoSystem->updateAdaptiveBlockSize(cands1.size() + cands2.size() +
                                     cands3.size());
  std::random_shuffle(cands1.begin(), cands1.end());
  std::random_shuffle(cands2.begin(), cands2.end());
  std::random_shuffle(cands3.begin(), cands3.end());
  int i1 = 0, i2 = 0, i3 = 0, i = 0;
  int &curPointId = dsoSystem->curPointId;
  while (i < settingInterestPointsUsed) {
    if (i1 < int(cands1.size())) {
      interestPoints[curPointId++] =
          std::make_unique<InterestPoint>(cands1[i1++]);
      i++;
    }
    if (i2 < int(cands2.size()) && i < settingInterestPointsUsed) {
      interestPoints[curPointId++] =
          std::make_unique<InterestPoint>(cands2[i2++]);
      i++;
    }
    if (i3 < int(cands3.size()) && i < settingInterestPointsUsed) {
      interestPoints[curPointId++] =
          std::make_unique<InterestPoint>(cands3[i3++]);
      i++;
    }
    if (i1 == int(cands1.size()) && i2 == int(cands2.size()) &&
        i3 == int(cands3.size()))
      break;
  }

#ifdef DEBUG
  char msg[500];
  sprintf(msg,
          "total pnt = %lu, 1's = %lu, 2's = %lu, 3's = %lu; block size = %i",
          cands1.size() + cands2.size() + cands3.size(), cands1.size(),
          cands2.size(), cands3.size(), dsoSystem->adaptiveBlockSize);
  cands1.resize(i1);
  cands2.resize(i2);
  cands3.resize(i3);
  frameWithPoints = frameColored.clone();
  cv::putText(frameWithPoints, msg, cv::Point(200, 200),
              cv::FONT_HERSHEY_SIMPLEX, 1, CV_BLACK, 3);
  for (cv::Point p : cands1)
    putDot(frameWithPoints, p, CV_GREEN);
  for (cv::Point p : cands2)
    putDot(frameWithPoints, p, CV_BLUE);
  for (cv::Point p : cands3)
    putDot(frameWithPoints, p, CV_RED);
#endif
}

int KeyFrame::getId() const { return frameId; }

int KeyFrame::getCols() const { return frame.cols; }

int KeyFrame::getRows() const { return frame.rows; }

} // namespace fishdso
