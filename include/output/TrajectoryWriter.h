#ifndef INCLUDE_TRAJECTORYWRITER
#define INCLUDE_TRAJECTORYWRITER

#include "output/DsoObserver.h"

namespace mdso {

class TrajectoryWriter : public DsoObserver {
public:
  virtual ~TrajectoryWriter();

  void newBaseFrame(const KeyFrame &baseFrame) override;
  void keyFramesMarginalized(const KeyFrame *marginalized[], int size) override;
  void destructed(const KeyFrame *lastKeyFrames[], int size) override;

  void outputModeFrameToWorld(const SE3 &frameToBody);
  void saveTimestamps(const fs::path &timestampsFile) const;
  const StdVector<SE3> &writtenFrameToWorld() const {
    return mWrittenFrameToWorld;
  }

private:
  virtual void addToPool(const KeyFrame &keyFrame) = 0;
  virtual void addToPool(const PreKeyFrame &frame) = 0;
  virtual PosesPool &frameToWorldPool() = 0;
  virtual const fs::path &outputFileName() const = 0;

  std::vector<Timestamp> writtenKfTs;
  std::vector<Timestamp> curKfTs;

  StdVector<SE3> mWrittenFrameToWorld;
  SE3 frameToBody;
};

} // namespace mdso

#endif
