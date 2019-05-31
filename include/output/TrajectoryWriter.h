#ifndef INCLUDE_TRAJECTORYWRITER
#define INCLUDE_TRAJECTORYWRITER

#include "output/DsoObserver.h"
#include <set>

namespace fishdso {

class TrajectoryWriter : public DsoObserver {
public:
  TrajectoryWriter(const std::string &outputDirectory,
                   const std::string &fileName,
                   const std::string &matrixFormFileName);

  void newKeyFrame(const KeyFrame *baseFrame);
  void keyFramesMarginalized(const std::vector<const KeyFrame *> &marginalized);
  void destructed(const std::vector<const KeyFrame *> &lastKeyFrames);

private:
  std::set<int> curKfNums;
  StdMap<int, SE3> frameToWorldPool;

  std::string outputFileName;
  std::string matrixFormOutputFileName;
};

} // namespace fishdso

#endif
