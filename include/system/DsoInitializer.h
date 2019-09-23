#ifndef INCLUDE_DSOINITIALIZER
#define INCLUDE_DSOINITIALIZER

#include "system/StereoMatcher.h"
#include <memory>
#include <opencv2/opencv.hpp>

namespace fishdso {

class CameraModel;

struct InitializedFrame {
  struct FrameEntry {
    FrameEntry(const cv::Mat &frame, Timestamp timestamp);

    cv::Mat frame;
    static_vector<std::pair<Vec2, double>,
                  Settings::KeyFrame::max_immaturePointsNum>
        depthedPoints;
    Timestamp timestamp;
  };

  InitializedFrame(cv::Mat frame[], Timestamp timestamps[], int size);

  SE3 thisToWorld;
  std::vector<FrameEntry> frames;
};

class DsoInitializer {
public:
  using InitializedVector =
      static_vector<InitializedFrame,
                    Settings::DsoInitializer::max_initializedFrames>;

  virtual ~DsoInitializer() {}

  // returns true if initialization is completed
  virtual bool addMultiFrame(const cv::Mat frames[],
                             Timestamp timestamps[]) = 0;

  virtual InitializedVector initialize() = 0;
};

} // namespace fishdso

#endif
