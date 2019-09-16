#ifndef INCLUDE_TRAJECTORYWRITERDSO
#define INCLUDE_TRAJECTORYWRITERDSO

#include "output/TrajectoryWriter.h"

namespace fishdso {

class TrajectoryWriterDso : public TrajectoryWriter {
public:
  TrajectoryWriterDso(const fs::path &outputDirectory,
                      const fs::path &fileName);

  void addToPool(const KeyFrame &keyFrame) override;
  void addToPool(const PreKeyFrame &frame) override;
  PosesPool &frameToWorldPool() override;
  const fs::path &outputFileName() override;

private:
  fs::path mOutputFileName;
  PosesPool mFrameToWorldPool;
};

} // namespace fishdso

#endif
