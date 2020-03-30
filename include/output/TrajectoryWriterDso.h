#ifndef INCLUDE_TRAJECTORYWRITERDSO
#define INCLUDE_TRAJECTORYWRITERDSO

#include "output/TrajectoryWriter.h"

namespace mdso {

class TrajectoryWriterDso : public TrajectoryWriter {
public:
  TrajectoryWriterDso(const fs::path &outputFname);

  void addToPool(const KeyFrame &keyFrame) override;
  void addToPool(const PreKeyFrame &frame) override;
  PosesPool &frameToWorldPool() override;
  const fs::path &outputFileName() const override;

private:
  fs::path mOutputFileName;
  PosesPool mFrameToWorldPool;
};

} // namespace mdso

#endif
