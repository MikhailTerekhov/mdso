#include "data/SingleCamProxyReader.h"

namespace mdso {

SingleCamProxyReader::Depths::Depths(std::unique_ptr<FrameDepths> depths,
                                     int realCamInd)
    : depths(std::move(depths))
    , realCamInd(realCamInd) {}

std::optional<double>
SingleCamProxyReader::Depths::depth(int camInd, const Vec2 &point) const {
  CHECK_EQ(camInd, 0);
  return depths->depth(realCamInd, point);
}

CameraBundle extractSingleCamera(const CameraBundle &cameraBundle, int camInd) {
  CHECK_GE(camInd, 0);
  CHECK_LT(camInd, cameraBundle.bundle.size());
  SE3 id;
  CameraModel cameraModel = cameraBundle.bundle[camInd].cam;
  return CameraBundle(&id, &cameraModel, 1);
}

SingleCamProxyReader::SingleCamProxyReader(
    std::unique_ptr<DatasetReader> newDatasetReader, int newCamInd)
    : datasetReader(std::move(newDatasetReader))
    , camInd(newCamInd)
    , mCam(extractSingleCamera(datasetReader->cam(), newCamInd))
    , frameToBody(datasetReader->cam().bundle[newCamInd].thisToBody) {}

int SingleCamProxyReader::numFrames() const {
  return datasetReader->numFrames();
}

int SingleCamProxyReader::firstTimestampToInd(Timestamp timestamp) const {
  return datasetReader->firstTimestampToInd(timestamp);
}

std::vector<Timestamp>
SingleCamProxyReader::timestampsFromInd(int frameInd) const {
  auto ts = datasetReader->timestampsFromInd(frameInd);
  return std::vector(1, ts[camInd]);
}

std::vector<DatasetReader::FrameEntry>
SingleCamProxyReader::frame(int frameInd) const {
  auto f = datasetReader->frame(frameInd);
  return std::vector<FrameEntry>(1, f[camInd]);
}

CameraBundle SingleCamProxyReader::cam() const { return mCam; }

std::unique_ptr<FrameDepths> SingleCamProxyReader::depths(int frameInd) const {
  auto depths = datasetReader->depths(frameInd);
  return std::unique_ptr<FrameDepths>(new Depths(std::move(depths), camInd));
}

bool SingleCamProxyReader::hasFrameToWorld(int frameInd) const {
  return datasetReader->hasFrameToWorld(frameInd);
}

SE3 SingleCamProxyReader::frameToWorld(int frameInd) const {
  SE3 bodyToWorld = datasetReader->frameToWorld(frameInd);
  return bodyToWorld * frameToBody;
}

} // namespace mdso
