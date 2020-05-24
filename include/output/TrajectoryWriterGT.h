#ifndef INCLUDE_TRAJECTORYWRITERGT
#define INCLUDE_TRAJECTORYWRITERGT

#include "data/DatasetReader.h"
#include "output/TrajectoryWriter.h"

namespace mdso {

class TrajectoryWriterGT : public TrajectoryWriter {
public:
  TrajectoryWriterGT(const DatasetReader *datasetReader,
                     const fs::path &outputDirectory, const fs::path &fileName);

  void setCamToBody(const SE3 &newCamToBody);

private:
  void addToPool(const KeyFrame &keyFrame) override;
  void addToPool(const PreKeyFrame &frame) override;
  PosesPool &frameToWorldPool() override;
  const fs::path &outputFileName() const override;

  void addToPoolByTimestamp(Timestamp ts);

  fs::path mOutputFileName;
  const DatasetReader *datasetReader;
  PosesPool frameToWorldGTPool;
  SE3 camToBody;
};

} // namespace mdso

#endif
