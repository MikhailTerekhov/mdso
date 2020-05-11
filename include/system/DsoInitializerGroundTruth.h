#ifndef INCLUDE_DSOINITIALIZERGROUNDTRUTH
#define INCLUDE_DSOINITIALIZERGROUNDTRUTH

#include "data/DatasetReader.h"
#include "system/DsoInitializer.h"

namespace mdso {

class DsoInitializerGroundTruth : public DsoInitializer {
public:
  DsoInitializerGroundTruth(const DatasetReader *datasetReader,
                            const InitializerGroundTruthSettings &settings);

  bool addMultiFrame(const cv::Mat newFrames[],
                     Timestamp newTimestamps[]) override;
  InitializedVector initialize() override;

private:
  void setFrame(const cv::Mat newFrames[], Timestamp newTimestamps[]);

  static constexpr int numInitializedFrames = 2;

  const DatasetReader *datasetReader;
  int numCams;
  int numSkippedFrames = -1;
  InitializedVector initializedVector;
  InitializerGroundTruthSettings settings;
};

} // namespace mdso

#endif
