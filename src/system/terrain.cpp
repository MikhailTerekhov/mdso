#include "system/terrain.h"

namespace fishdso {

Terrain::Terrain(CameraModel *cam, const stdvectorVec2 &points,
                 const std::vector<double> &depths)
    : depths(depths), triang(points) {
  if (points.size() != depths.size())
    throw std::runtime_error("bad Terrain initialization!");
  refRays.resize(points.size());
  for (int i = 0; i < int(points.size()); ++i) {
    Vec3 ray = cam->unmap(points[i].data());
    ray.normalize();
    ray *= depths[i];
    refRays[i] = ray;
  }

  debugOut = false;
}

bool Terrain::hasInterpolatedDepth(Vec2 p) {
  return !triang.isIncidentToBoundary(triang.enclosingTriangle(p));
}

bool Terrain::operator()(Vec2 p, double &resDepth) {
  auto tri = triang.enclosingTriangle(p);
  if (tri == nullptr || triang.isIncidentToBoundary(tri))
    return false;

  Vec3 depths;
  for (int i = 0; i < 3; ++i) {
    int curInd = tri->vert[i]->index;
    depths[i] = refRays[curInd].norm();
  }

  Mat33 A;
  for (int i = 0; i < 3; ++i)
    A.block<1, 2>(i, 0) = tri->vert[i]->pos.transpose();
  A.block<3, 1>(0, 2) = Vec3::Ones();
  Vec3 coeffs = A.fullPivHouseholderQr().solve(depths);
  resDepth = coeffs[0] * p[0] + coeffs[1] * p[1] + coeffs[2];
  return true;
}

void Terrain::draw(cv::Mat &img, cv::Scalar edgeCol) {
  triang.draw(img, edgeCol);
}

void Terrain::drawDensePlainDepths(cv::Mat &img, double minDepth,
                                   double maxDepth) {
  if (img.type() != CV_8UC3) {
    std::cerr << "wrong img type in drawDensePlainDepths" << std::endl;
    throw std::runtime_error("wrong img type in drawDensePlainDepths");
  }
  for (int y = 0; y < img.rows; ++y)
    for (int x = 0; x < img.cols; ++x) {
      double depth;
      if (operator()(Vec2(double(x), double(y)), depth))
        img.at<cv::Vec3b>(y, x) =
            toCvVec3bDummy(depthCol(depth, minDepth, maxDepth));
    }
}

void Terrain::drawCurved(CameraModel *cam, cv::Mat &img, cv::Scalar edgeCol) {
  triang.drawCurved(cam, img, edgeCol);
}

} // namespace fishdso
