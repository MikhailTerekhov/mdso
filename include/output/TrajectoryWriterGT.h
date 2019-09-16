#ifndef INCLUDE_TRAJECTORYWRITERGT
#define INCLUDE_TRAJECTORYWRITERGT

#include "output/TrajectoryWriter.h"

namespace fishdso {

class TrajectoryWriterGT : public TrajectoryWriter {
public:
  TrajectoryWriterGT(const SE3 _frameToWorldGT[], Timestamp _timestamps[],
                     int size, const fs::path &outputDirectory,
                     const fs::path &fileName);

  void addToPool(const KeyFrame &keyFrame) override;
  void addToPool(const PreKeyFrame &frame) override;
  PosesPool &frameToWorldPool() override;
  const fs::path &outputFileName() override;

private:
  void addToPoolByTimestamp(Timestamp ts);

  StdVector<SE3> frameToWorldGT;
  std::vector<Timestamp> timestamps;
  fs::path mOutputFileName;
  PosesPool frameToWorldGTPool;
};

} // namespace fishdso

#endif
