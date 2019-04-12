#ifndef INCLUDE_TRAJECTORYWRITER
#define INCLUDE_TRAJECTORYWRITER

#include "output/DsoObserver.h"
#include <optional>

namespace fishdso {

class TrajectoryWriter : public DsoObserver {
public:
  TrajectoryWriter(const std::string &outputDirectory,
                   const std::string &fileName);

  void keyFramesMarginalized(const std::vector<const KeyFrame *> &marginalized);
  void destructed(const std::vector<const KeyFrame *> &lastKeyFrames);

private:
  std::string outputFileName;
};

} // namespace fishdso

#endif
