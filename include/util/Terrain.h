#ifndef INCLUDE_TERRAIN
#define INCLUDE_TERRAIN

#include "system/CameraModel.h"
#include "util/Triangulation.h"

namespace mdso {

class Terrain {
  using Vertex = Triangulation::Vertex;
  using Edge = Triangulation::Edge;
  using Triangle = Triangulation::Triangle;

public:
  Terrain(const CameraModel *cam, const StdVector<Vec2> &points,
          const std::vector<double> &depths,
          const Settings::Triangulation &triangulationSettings = {});

  bool hasInterpolatedDepth(const Vec2 &p);
  bool hasInterpolatedDepth(const Vec2 &p, double &nodeProximity);
  bool operator()(const Vec2 &p, double &resDepth) const;

  void draw(cv::Mat &img, cv::Scalar edgeCol);
  void draw(cv::Mat &img, cv::Scalar edgeCol, int thickness);
  void drawDensePlainDepths(cv::Mat &img, double minDepth, double maxDepth);
  void drawCurved(CameraModel *cam, cv::Mat &img, cv::Scalar edgeCol);

  bool debugOut;

private:
  std::vector<double> depths;
  Triangulation triang;
  std::vector<Vec3> refRays;
};

} // namespace mdso

#endif
