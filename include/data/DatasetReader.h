#ifndef INCLUDE_DATASETREADER
#define INCLUDE_DATASETREADER

#include "system/CameraBundle.h"
#include "util/types.h"
#include <memory>
#include <optional>

namespace mdso {

class FrameDepths {
public:
  virtual std::optional<double> depth(int camInd, const Vec2 &point) const = 0;
};

class DatasetReader {
public:
  struct FrameEntry {
    cv::Mat3b frame;
    Timestamp timestamp;
  };

  virtual ~DatasetReader() = 0;

  virtual int numFrames() const = 0;
  virtual std::vector<FrameEntry> frame(int frameInd) const = 0;
  virtual CameraBundle cam() const = 0;
  virtual std::unique_ptr<FrameDepths> depths(int frameInd) const = 0;
  virtual std::optional<SE3> frameToWorld(int frameInd) const = 0;
};

} // namespace mdso

#endif