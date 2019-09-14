#ifndef INCLUDE_TRAJECTORYWRITERGT
#define INCLUDE_TRAJECTORYWRITERGT

#include "output/TrajectoryWriter.h"

namespace fishdso {

class TrajectoryWriterGT : public TrajectoryWriter {
public:
  TrajectoryWriterGT(const SE3 _frameToWorldGT[], Timestamp _timestamps[],
                     int size, const std::string &outputDirectory,
                     const std::string &fileName);

  void addToPool(const KeyFrame &keyFrame) override;
  void addToPool(const PreKeyFrame &frame) override;
  PosesPool &frameToWorldPool() override;
  const std::string &outputFileName() override;

private:
  void addToPoolByTimestamp(Timestamp ts);

  StdVector<SE3> frameToWorldGT;
  std::vector<Timestamp> timestamps;
  std::string mOutputFileName;
  PosesPool frameToWorldGTPool;
};

} // namespace fishdso

#endif
