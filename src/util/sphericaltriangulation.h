#pragma once

#include "system/cameramodel.h"
#include "util/triangulation.h"
#include "util/types.h"

namespace fishdso {

class SphericalTriangulation {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  struct TrihedralSector {
    Vec3 *rays[3];
  };

  SphericalTriangulation(const std::vector<Vec3> &rays);

  TrihedralSector *enclosingSector(Vec3 ray);

  void checkAllSectors(Vec3 ray, CameraModel *cam, cv::Mat &img);

  void fillUncovered(cv::Mat &img, CameraModel *cam, cv::Scalar fillCol);

  cv::Mat drawTangentTri(int imWidth, int imHeight);
  void draw(cv::Mat &img, CameraModel *cam, cv::Scalar edgeCol);

private:
  bool isInConvexDummy(Vec3 ray);

  Triangulation tangentTriang;
  std::vector<Vec3> _rays;
  std::map<const Triangulation::Triangle *, TrihedralSector> _sectors;
};

} // namespace fishdso
