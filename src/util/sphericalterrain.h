#pragma once

#include "system/cameramodel.h"
#include "util/sphericaltriangulation.h"
#include <opencv2/opencv.hpp>

namespace fishdso {

class SphericalTerrain {
public:
  SphericalTerrain(const std::vector<Vec3> &depthedRays);

  bool operator()(Vec3 direction, double &resDepth);

  void checkAllSectors(Vec3 ray, CameraModel *cam, cv::Mat &img);

  void draw(cv::Mat &img, CameraModel *cam, cv::Scalar edgeCol, double minDepth,
            double maxDepth);

  void fillUncovered(cv::Mat &img, CameraModel *cam, cv::Scalar fillCol);
  cv::Mat drawTangentTri(int imWidth, int imHeight);

private:
  void drawDenseTriangle(cv::Mat &img, CameraModel *cam,
                         SphericalTriangulation::TrihedralSector *sec,
                         double minDepth, double maxDepth);

  SphericalTriangulation triang;
};

} // namespace fishdso