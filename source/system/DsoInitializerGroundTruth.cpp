#include "system/DsoInitializerGroundTruth.h"
#include "util/PixelSelector.h"

namespace mdso {

DsoInitializerGroundTruth::DsoInitializerGroundTruth(
    const DatasetReader *datasetReader,
    const InitializerGroundTruthSettings &settings)
    : datasetReader(datasetReader)
    , numCams(datasetReader->cam().bundle.size())
    , settings(settings) {
  initializedVector.reserve(numInitializedFrames);
}

void DsoInitializerGroundTruth::setFrame(const cv::Mat newFrames[],
                                         Timestamp newTimestamps[]) {
  initializedVector.emplace_back(newFrames, newTimestamps, numCams);
  int frameInd = datasetReader->firstTimestampToInd(newTimestamps[0]);
  initializedVector.back().thisToWorld = datasetReader->frameToWorld(frameInd);
}

bool DsoInitializerGroundTruth::addMultiFrame(const cv::Mat newFrames[],
                                              Timestamp newTimestamps[]) {
  if (numSkippedFrames == -1) {
    setFrame(newFrames, newTimestamps);
  } else if (numSkippedFrames == settings.intializer.firstFramesSkip) {
    setFrame(newFrames, newTimestamps);
    return true;
  }
  numSkippedFrames++;
  return false;
}

DsoInitializer::InitializedVector DsoInitializerGroundTruth::initialize() {
  CHECK_EQ(initializedVector.size(), numInitializedFrames)
      << "intialize was called before the frames were set";

  int pointsPerFrame = settings.keyFrame.immaturePointsNum() / numCams;
  std::vector<PixelSelector> pixelSelectors;
  pixelSelectors.reserve(numCams);
  for (int camInd = 0; camInd < numCams; ++camInd) {
    pixelSelectors.emplace_back(settings.pixelSelector);
    pixelSelectors.back().initialize(initializedVector[0].frames[camInd].frame,
                                     pointsPerFrame);
  }
  for (int frameInd = 0; frameInd < numInitializedFrames; ++frameInd) {
    InitializedFrame &initializedMultiFrame = initializedVector[frameInd];
    int globalFrameInd = datasetReader->firstTimestampToInd(
        initializedMultiFrame.frames[0].timestamp);
    auto depths = datasetReader->depths(globalFrameInd);
    for (int camInd = 0; camInd < numCams; ++camInd) {
      cv::Mat3b frame = initializedMultiFrame.frames[camInd].frame;
      cv::Mat1d gradX, gradY, gradNorm;
      grad(cvtBgrToGray(frame), gradX, gradY, gradNorm);
      PixelSelector::PointVector points =
          pixelSelectors[camInd].select(frame, gradNorm, pointsPerFrame);
      for (const cv::Point &pCv : points) {
        Vec2 p = toVec2(pCv);
        if (auto d = depths->depth(camInd, p); d)
          initializedMultiFrame.frames[camInd].depthedPoints.emplace_back(p,
                                                                          *d);
      }
      int totalInitPoints =
          initializedMultiFrame.frames[camInd].depthedPoints.size();
      LOG(INFO) << "frame #" << frameInd << " cam #" << camInd
                << " initialized " << totalInitPoints << " / " << points.size();
      CHECK_GT(totalInitPoints, 0);
    }
  }

  return initializedVector;
}

} // namespace mdso