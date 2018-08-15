#pragma once

#include "dsosystem.h"
#include "interestpoint.h"
#include <map>
#include <memory>
#include <opencv2/core.hpp>

namespace fishdso {

class DsoSystem;

struct KeyFrame {
  friend class DsoSystem;

  KeyFrame(int frameId, cv::Mat frameColored, DsoSystem *dsoSystem);

  int getId() const;
  int getCols() const;
  int getRows() const;

  void selectPoints();

  int frameId;
  cv::Mat frame;
  cv::Mat frameColored;
  cv::Mat gradX, gradY, gradNorm;
  DsoSystem *dsoSystem;
  std::map<int, std::unique_ptr<InterestPoint>> interestPoints;

#ifdef DEBUG
  cv::Mat frameWithPoints;
#endif
};

} // namespace fishdso
