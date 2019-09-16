#ifndef INCLUDE_TRAJECTORYWRITER
#define INCLUDE_TRAJECTORYWRITER

#include "output/DsoObserver.h"

namespace fishdso {

class TrajectoryWriter : public DsoObserver {
public:
  virtual ~TrajectoryWriter();

  void newBaseFrame(const KeyFrame &baseFrame) override;
  void keyFramesMarginalized(const KeyFrame *marginalized[], int size) override;
  void destructed(const KeyFrame *lastKeyFrames[], int size) override;

  virtual void addToPool(const KeyFrame &keyFrame) = 0;
  virtual void addToPool(const PreKeyFrame &frame) = 0;
  virtual PosesPool &frameToWorldPool() = 0;
  virtual const fs::path &outputFileName() = 0;

private:
  std::vector<Timestamp> curKfTs;
};

} // namespace fishdso

#endif
