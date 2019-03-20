#ifndef INCLUDE_SPHERICAL_TERRAIN
#define INCLUDE_SPHERICAL_TERRAIN

#include "system/CameraModel.h"
#include "util/SphericalTriangulation.h"
#include <opencv2/opencv.hpp>

namespace fishdso {

class SphericalTerrain {
public:
  SphericalTerrain(const std::vector<Vec3> &depthedRays,
                   const Settings::Triangulation &triangulationSettings = {});

  bool operator()(Vec3 direction, double &resDepth);

  void checkAllSectors(Vec3 ray, CameraModel *cam, cv::Mat &img);

  void draw(cv::Mat &img, CameraModel *cam, cv::Scalar edgeCol, double minDepth,
            double maxDepth);

  void fillUncovered(cv::Mat &img, CameraModel *cam, cv::Scalar fillCol);
  cv::Mat drawTangentTri(int imWidth, int imHeight, cv::Scalar bgCol,
                         cv::Scalar edgeCol);

private:
  void drawDenseTriangle(cv::Mat &img, CameraModel *cam,
                         SphericalTriangulation::TrihedralSector *sec,
                         double minDepth, double maxDepth);

  SphericalTriangulation triang;

  Settings::Triangulation settings;
};

} // namespace fishdso

#endif
