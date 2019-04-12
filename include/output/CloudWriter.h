#ifndef INCLUDE_CLOUDWRITER
#define INCLUDE_CLOUDWRITER

#include "output/DsoObserver.h"
#include "util/PlyHolder.h"

namespace fishdso {

class CloudWriter : public DsoObserver {
public:
  CloudWriter(CameraModel *cam, const std::string &outputDirectory,
                 const std::string &fileName);
  void keyFramesMarginalized(const std::vector<const KeyFrame *> &marginalized);
  void destructed(const std::vector<const KeyFrame *> &lastKeyFrames);

private:
  CameraModel *cam;
  std::string outputDirectory;
  PlyHolder cloudHolder;
};

} // namespace fishdso

#endif
