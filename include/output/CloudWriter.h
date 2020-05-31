#ifndef INCLUDE_CLOUDWRITER
#define INCLUDE_CLOUDWRITER

#include "output/DsoObserver.h"
#include "util/PlyHolder.h"

namespace mdso {

class CloudWriter : public DsoObserver {
public:
  CloudWriter(CameraBundle *cam, const fs::path &outputDirectory,
              const fs::path &fileName, bool newOutputStddev = true);
  void keyFramesMarginalized(const KeyFrame *marginalized[], int size) override;
  void destructed(const KeyFrame *lastKeyFrames[], int size) override;

private:
  CameraBundle *cam;
  fs::path outputDirectory;
  PlyHolder cloudHolder;
  bool outputStddev;
};

} // namespace mdso

#endif
