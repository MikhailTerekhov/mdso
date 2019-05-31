#ifndef INCLUDE_TRAJECTORYWRITERGT
#define INCLUDE_TRAJECTORYWRITERGT

#include "output/DsoObserver.h"
#include "util/Sim3Aligner.h"

namespace fishdso {

class TrajectoryWriterGT : public DsoObserver {
public:
  TrajectoryWriterGT(const StdVector<SE3> &worldToFrameUnalignedGT,
                     const std::string &outputDirectory,
                     const std::string &fileName,
                     const std::string &matrixFormFileName);

  void initialized(const std::vector<const KeyFrame *> &marginalized);
  void keyFramesMarginalized(const std::vector<const KeyFrame *> &marginalized);
  void destructed(const std::vector<const KeyFrame *> &lastKeyFrames);

private:
  StdVector<SE3> worldToFrameGT;
  StdVector<SE3> worldToFrameUnalignedGT;
  std::string outputFileName;
  std::string matrixFormOutputFileName;
  std::unique_ptr<Sim3Aligner> sim3Aligner;
};

} // namespace fishdso

#endif
