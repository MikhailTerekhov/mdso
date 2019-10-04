#ifndef INCLUDE_TRAJECTORYWRITERPREDICT
#define INCLUDE_TRAJECTORYWRITERPREDICT

#include "output/TrajectoryWriter.h"

namespace mdso {

class TrajectoryWriterPredict : public TrajectoryWriter {
public:
  TrajectoryWriterPredict(const fs::path &outputDirectory,
                          const fs::path &fileName);

  void addToPool(const KeyFrame &keyFrame) override;
  void addToPool(const PreKeyFrame &frame) override;
  PosesPool &frameToWorldPool() override;
  const fs::path &outputFileName() override;

private:
  fs::path mOutputFileName;
  PosesPool mFrameToWorldPool;
};

} // namespace mdso

#endif
