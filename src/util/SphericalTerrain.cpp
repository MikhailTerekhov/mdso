#include "util/SphericalTerrain.h"
#include "util/defs.h"

namespace fishdso {

SphericalTerrain::SphericalTerrain(const std::vector<Vec3> &depthedRays)
    : triang(depthedRays) {}

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
                            double minDepth, double maxDepth) {
  triang.draw(img, cam, edgeCol);

  SphericalTriangulation::TrihedralSector *sec =
      triang.enclosingSector(cam->unmap(Vec2(850.0, 250.0).data()));
  //  if (sec == nullptr)
  //    std::cout << "no tri found" << std::endl;
  //  else {
  //    std::cout << "enclosing tri verts = " << std::endl;
  //    for (Vec3 *r : sec->rays)
  //      std::cout << cam->map(r->data()).transpose() << std::endl;

  //    std::cout << "and without mapping:" << std::endl;
  //    for (Vec3 *r : sec->rays)
  //      std::cout << r->transpose() << std::endl;
  //    drawDenseTriangle(img, cam, sec, 0., 1.);
  //  }
}

void SphericalTerrain::fillUncovered(cv::Mat &img, CameraModel *cam,
                                     cv::Scalar fillCol) {
  triang.fillUncovered(img, cam, fillCol);
}

cv::Mat SphericalTerrain::drawTangentTri(int imWidth, int imHeight) {
  return triang.drawTangentTri(imWidth, imHeight);
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
