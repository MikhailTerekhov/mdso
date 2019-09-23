#ifndef INCLUDE_TRAJECTORYWRITER
#define INCLUDE_TRAJECTORYWRITER

#include "output/DsoObserver.h"
#include <set>

namespace fishdso {

class TrajectoryWriter : public DsoObserver {
public:
  TrajectoryWriter(const std::string &outputDirectory,
                   const std::string &fileName);

  void newBaseFrame(const KeyFrame &baseFrame) override;
  void keyFramesMarginalized(const KeyFrame *marginalized[], int size) override;
  void destructed(const KeyFrame *lastKeyFrames[], int size) override;

private:
  std::vector<Timestamp> curKfTs;
  PosesPool frameToWorldPool;

  std::string outputFileName;
};

} // namespace fishdso

#endif
