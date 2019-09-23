#ifndef INCLUDE_CLOUDWRITER
#define INCLUDE_CLOUDWRITER

#include "output/DsoObserver.h"
#include "util/PlyHolder.h"

namespace fishdso {

class CloudWriter : public DsoObserver {
public:
  CloudWriter(CameraBundle *cam, const std::string &outputDirectory,
              const std::string &fileName);
  void keyFramesMarginalized(const KeyFrame *marginalized[], int size) override;
  void destructed(const KeyFrame *lastKeyFrames[], int size) override;

private:
  CameraBundle *cam;
  std::string outputDirectory;
  PlyHolder cloudHolder;
};

} // namespace fishdso

#endif
