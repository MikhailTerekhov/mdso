#ifndef INCLUDE_TRAJECTORYWRITERDSO
#define INCLUDE_TRAJECTORYWRITERDSO

#include "output/TrajectoryWriter.h"

namespace fishdso {

class TrajectoryWriterDso : public TrajectoryWriter {
public:
  TrajectoryWriterDso(const std::string &outputDirectory,
                      const std::string &fileName);

  void addToPool(const KeyFrame &keyFrame) override;
  void addToPool(const PreKeyFrame &frame) override;
  PosesPool &frameToWorldPool() override;
  const std::string &outputFileName() override;

private:
  std::string mOutputFileName;
  PosesPool mFrameToWorldPool;
};

} // namespace fishdso

#endif
