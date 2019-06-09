#ifndef INCLUDE_CLOUDWRITERGT
#define INCLUDE_CLOUDWRITERGT

#include "output/DsoObserver.h"
#include "util/PlyHolder.h"
#include "util/Sim3Aligner.h"
#include "util/types.h"

namespace fishdso {

class CloudWriterGT : public DsoObserver {
public:
  CloudWriterGT(const StdVector<SE3> &worldToFrameGT,
                const std::vector<std::vector<Vec3>> &pointsInFrameGT,
                const std::vector<std::vector<cv::Vec3b>> &colors,
                const std::string &outputDirectory,
                const std::string &fileName);

  void initialized(const std::vector<const KeyFrame *> &initializedKFs);
  void keyFramesMarginalized(const std::vector<const KeyFrame *> &marginalized);
  void destructed(const std::vector<const KeyFrame *> &lastKeyFrames);

private:
  StdVector<SE3> worldToFrameGT;
  std::vector<std::vector<Vec3>> pointsInFrameGT;
  std::vector<std::vector<cv::Vec3b>> colors;
  PlyHolder cloudHolder;
  std::unique_ptr<Sim3Aligner> sim3Aligner;
};

} // namespace fishdso

#endif
