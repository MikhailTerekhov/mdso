#ifndef INCLUDE_CLOUDWRITERGT
#define INCLUDE_CLOUDWRITERGT

#include "output/DsoObserver.h"
#include "util/PlyHolder.h"
#include "util/Sim3Aligner.h"
#include "util/types.h"

namespace fishdso {

class CloudWriterGT : public DsoObserver {
public:
  CloudWriterGT(SE3 frameToWorldGT[], Timestamp timestamps[],
                std::vector<Vec3> pointsInFrameGT[],
                std::vector<cv::Vec3b> colors[], int size,
                const std::string &outputDirectory,
                const std::string &fileName);

  void initialized(const KeyFrame *initializedKFs[], int size) override;
  void keyFramesMarginalized(const KeyFrame *marginalized[], int size) override;
  void destructed(const KeyFrame *lastKeyFrames[], int size) override;

private:
  int findInd(Timestamp timestamp);
  
  std::vector<Timestamp> timestamps;
  StdVector<SE3> frameToWorldGT;
  std::vector<std::vector<Vec3>> pointsInFrameGT;
  std::vector<std::vector<cv::Vec3b>> colors;
  PlyHolder cloudHolder;
  std::unique_ptr<Sim3Aligner> sim3Aligner;
};

} // namespace fishdso

#endif
