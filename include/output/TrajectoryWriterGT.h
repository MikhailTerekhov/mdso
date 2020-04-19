#ifndef INCLUDE_TRAJECTORYWRITERGT
#define INCLUDE_TRAJECTORYWRITERGT

#include "output/TrajectoryWriter.h"

namespace mdso {

class TrajectoryWriterGT : public TrajectoryWriter {
public:
  TrajectoryWriterGT(const SE3 _frameToWorldGT[], Timestamp _timestamps[],
                     int size, const fs::path &outputDirectory,
                     const fs::path &fileName);

private:
  void addToPool(const KeyFrame &keyFrame) override;
  void addToPool(const PreKeyFrame &frame) override;
  PosesPool &frameToWorldPool() override;
  const fs::path &outputFileName() const override;

  void addToPoolByTimestamp(Timestamp ts);

  StdVector<SE3> frameToWorldGT;
  std::vector<Timestamp> timestamps;
  fs::path mOutputFileName;
  PosesPool frameToWorldGTPool;
};

} // namespace mdso

#endif
