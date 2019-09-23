#ifndef INCLUDE_TRAJECTORYWRITERGT
#define INCLUDE_TRAJECTORYWRITERGT

#include "output/DsoObserver.h"
#include "util/Sim3Aligner.h"

namespace fishdso {

class TrajectoryWriterGT : public DsoObserver {
public:
  TrajectoryWriterGT(const SE3 frameToWorldGT[], Timestamp timestamps[],
                     int size, const std::string &outputDirectory,
                     const std::string &fileName);

  void newBaseFrame(const KeyFrame &baseFrame) override;
  void keyFramesMarginalized(const KeyFrame *marginalized[], int size) override;
  void destructed(const KeyFrame *lastKeyFrames[], int size) override;

private:
  std::vector<Timestamp> curKfTs;
  PosesPool frameToWorldGTPool;

  std::string outputFileName;
};

} // namespace fishdso

#endif
