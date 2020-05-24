#ifndef INCLUDE_SINGLECAMPROXYREADER
#define INCLUDE_SINGLECAMPROXYREADER

#include "data/DatasetReader.h"

namespace mdso {

class SingleCamProxyReader : public DatasetReader {
public:
  class Depths : public FrameDepths {
  public:
    Depths(std::unique_ptr<FrameDepths> depths, int realCamInd);

    std::optional<double> depth(int camInd, const Vec2 &point) const override;

  private:
    std::unique_ptr<FrameDepths> depths;
    int realCamInd;
  };

  SingleCamProxyReader(std::unique_ptr<DatasetReader> datasetReader,
                       int newCamInd);

  int numFrames() const override;
  int firstTimestampToInd(Timestamp timestamp) const override;
  std::vector<Timestamp> timestampsFromInd(int frameInd) const override;
  std::vector<FrameEntry> frame(int frameInd) const override;
  CameraBundle cam() const override;
  std::unique_ptr<FrameDepths> depths(int frameInd) const override;
  bool hasFrameToWorld(int frameInd) const override;
  SE3 frameToWorld(int frameInd) const override;

private:
  std::unique_ptr<DatasetReader> datasetReader;
  int camInd;
  CameraBundle mCam;
  SE3 frameToBody;
};

} // namespace mdso

#endif
