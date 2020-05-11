#include "system/DsoInitializer.h"

namespace mdso {

InitializedFrame::FrameEntry::FrameEntry(const cv::Mat &frame,
                                         Timestamp timestamp)
    : frame(frame.clone())
    , timestamp(timestamp) {}

InitializedFrame::InitializedFrame(const cv::Mat frame[],
                                   Timestamp timestamps[], int size) {
  frames.reserve(size);
  for (int i = 0; i < size; ++i)
    frames.emplace_back(frame[i], timestamps[i]);
}

} // namespace mdso
