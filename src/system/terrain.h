#include "system/cameramodel.h"
#include "util/triangulation.h"

namespace fishdso {

class Terrain {
  using Vertex = Triangulation::Vertex;
  using Edge = Triangulation::Edge;
  using Triangle = Triangulation::Triangle;

public:
  Terrain(CameraModel *cam, const stdvectorVec2 &points,
          const std::vector<double> &depths);

  bool hasInterpolatedDepth(Vec2 p);
  bool operator()(Vec2 p, double &resDepth);

  void draw(cv::Mat &img, cv::Scalar edgeCol);
  void drawDensePlainDepths(cv::Mat &img, double minDepth, double maxDepth);
  void drawCurved(CameraModel *cam, cv::Mat &img, cv::Scalar edgeCol);

  bool debugOut;

private:
  std::vector<double> depths;
  Triangulation triang;
  std::vector<Vec3> refRays;
};

} // namespace fishdso
