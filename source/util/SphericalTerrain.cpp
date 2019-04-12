#include "util/SphericalTerrain.h"
#include "util/defs.h"

namespace fishdso {

SphericalTerrain::SphericalTerrain(
    const std::vector<Vec3> &depthedRays,
    const Settings::Triangulation &triangulationSettings)
    : triang(depthedRays, triangulationSettings) {}

bool SphericalTerrain::operator()(Vec3 direction, double &resDepth) {
  SphericalTriangulation::TrihedralSector *sec =
      triang.enclosingSector(direction);

  if (sec == nullptr)
    return false;

  Mat43 A;
  for (int i = 0; i < 3; ++i)
    A.block<3, 1>(0, i) = *sec->rays[i];
  A.block<1, 3>(3, 0) = Vec3::Ones().transpose();
  Mat44 Q = A.householderQr().householderQ();
  Vec4 plane = Q.col(3);
  double alpha = -plane[3] / plane.head<3>().dot(direction);
  resDepth = alpha * direction.norm();
  return true;
}

void SphericalTerrain::checkAllSectors(Vec3 ray, CameraModel *cam,
                                       cv::Mat &img) {
  triang.checkAllSectors(ray, cam, img);
}

void SphericalTerrain::draw(cv::Mat &img, CameraModel *cam, cv::Scalar edgeCol,
                            double minDepth, double maxDepth) const {
  triang.draw(img, cam, edgeCol);
}

void SphericalTerrain::fillUncovered(cv::Mat &img, CameraModel *cam,
                                     cv::Scalar fillCol) {
  triang.fillUncovered(img, cam, fillCol);
}

cv::Mat SphericalTerrain::drawTangentTri(int imWidth, int imHeight,
                                         cv::Scalar bgCol, cv::Scalar edgeCol) {
  return triang.drawTangentTri(imWidth, imHeight, bgCol, edgeCol);
}

void SphericalTerrain::drawDenseTriangle(
    cv::Mat &img, CameraModel *cam,
    SphericalTriangulation::TrihedralSector *sec, double minDepth,
    double maxDepth) {
  for (int y = 0; y < img.rows; y += 10)
    for (int x = 0; x < img.cols; x += 10) {
      Vec3 ray = cam->unmap(Vec2(double(x), double(y)).data());
      SphericalTriangulation::TrihedralSector *encl =
          triang.enclosingSector(ray);
      if (encl != sec)
        continue;
      double depth;
      if (operator()(ray, depth))
        putDot(img, cv::Point(x, y), depthCol(depth, minDepth, maxDepth));
    }
}

} // namespace fishdso
