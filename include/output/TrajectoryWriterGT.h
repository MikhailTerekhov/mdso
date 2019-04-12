#ifndef INCLUDE_TRAJECTORYWRITERGT
#define INCLUDE_TRAJECTORYWRITERGT

#include "output/DsoObserver.h"
#include "util/Sim3Aligner.h"

namespace fishdso {

class TrajectoryWriterGT : public DsoObserver {
public:
  TrajectoryWriterGT(const StdVector<SE3> &worldToFrameGT,
                     const std::string &outputDirectory,
                     const std::string &fileName);

  void initialized(const std::vector<const KeyFrame *> &marginalized);
  void keyFramesMarginalized(const std::vector<const KeyFrame *> &marginalized);
  void destructed(const std::vector<const KeyFrame *> &lastKeyFrames);

private:
  StdVector<SE3> worldToFrameGT;
  std::string outputFileName;
  std::unique_ptr<Sim3Aligner> sim3Aligner;
};

} // namespace fishdso

#endif
