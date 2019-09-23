#include "util/SphericalTriangulation.h"
#include "util/defs.h"
#include "util/geometry.h"
#include <glog/logging.h>

namespace mdso {

Vec2 stereographicProject(Vec3 ray) {
  static const Vec3 t(0.0, 0.0, 1.0);
  ray.normalize();
  double s = 2 / (1 + ray.dot(t));
  // std cout << "third component = " << (s * ray + (s - 1) * t)[2] <<
  // std::endl;
  return (s * ray + (s - 1) * t).head<2>();
}

double sectorBadness(SphericalTriangulation::TrihedralSector *sec) {
  double maxAngle = -std::numeric_limits<double>::infinity();
  for (int i = 0; i < 3; ++i)
    maxAngle =
        std::max(maxAngle, angle(*sec->rays[i], *sec->rays[(i + 1) % 3]));
  return maxAngle;
}

StdVector<Vec2> projectAll(const std::vector<Vec3> &rays) {
  StdVector<Vec2> result(rays.size());
  std::transform(rays.begin(), rays.end(), result.begin(),
                 stereographicProject);
  return result;
}

SphericalTriangulation::SphericalTriangulation(
    const std::vector<Vec3> &rays, const Settings::Triangulation &settings)
    : tangentTriang(projectAll(rays), settings)
    , _rays(rays) {
  for (auto tri : tangentTriang.triangles()) {
    for (int i = 0; i < 3; ++i)
      _sectors[tri].rays[i] = &_rays[tri->vert[i]->index];
  }
}

SphericalTriangulation::TrihedralSector *
SphericalTriangulation::enclosingSector(Vec3 ray) {
  std::vector<
      std::map<const Triangulation::Triangle *, TrihedralSector>::iterator>
      secIters;
  for (auto it = _sectors.begin(); it != _sectors.end(); ++it)
    if (isInSector(ray, it->second.rays))
      secIters.push_back(it);
  if (secIters.size() == 1) {
    return &secIters[0]->second;
  } else if (secIters.size() > 1) {
    // std::cout << "removing some sectors!" << std::endl;

    auto bestSecIter = *std::min_element(
        secIters.begin(), secIters.end(), [](auto si1, auto si2) {
          return sectorBadness(&si1->second) < sectorBadness(&si2->second);
        });
    for (auto si : secIters) {
      if (si != bestSecIter)
        _sectors.erase(si);
    }

    return &bestSecIter->second;
  }

  return nullptr;
}

void drawCurvedInternal(CameraModel *cam, Vec3 rayFrom, Vec3 rayTo,
                        cv::Mat &img, cv::Scalar edgeCol, int thickness) {
  constexpr int curveSectors = 50;
  constexpr int curvePoints = curveSectors + 1;

  rayFrom.normalize();
  rayTo.normalize();

  //  std::cout << "edge from " << cam->map(rayFrom.data()).transpose() << " to
  //  "
  //            << cam->map(rayTo.data()).transpose() << std::endl;

  double fullAngle = std::acos(rayFrom.dot(rayTo));
  Vec3 axis = rayFrom.cross(rayTo).normalized();
  cv::Point pnts[curvePoints];

  for (int it = 0; it < curvePoints; ++it) {
    double curAngle = fullAngle * it / curveSectors;
    SO3 curRot = SO3::exp(curAngle * axis);
    Vec3 rayCur = curRot * rayFrom;

    pnts[it] = toCvPoint(cam->map(rayCur.data()));
  }

  for (int it = 0; it < curveSectors; ++it)
    cv::line(img, pnts[it], pnts[it + 1], edgeCol, thickness);
}

void SphericalTriangulation::checkAllSectors(Vec3 ray, CameraModel *cam,
                                             cv::Mat &img) {
  static bool secDrawn = false;
  std::vector<TrihedralSector *> sec;
  for (auto &triSec : _sectors)
    if (isInSector(ray, triSec.second.rays))
      sec.push_back(&triSec.second);
  if (sec.size() > 1) {
    LOG(INFO) << sec.size() << " sectors pnt!" << std::endl;
    LOG(INFO) << "p = " << cam->map(ray.data()).transpose() << std::endl;
    putDot(img, toCvPoint(cam->map(ray.data())), CV_BLACK);

    int i = 0;
    for (auto s : sec) {
      LOG(INFO) << "sec " << i++ << " mapped" << std::endl;
      for (auto r : s->rays)
        LOG(INFO) << cam->map(r->data()).transpose() << std::endl;
      LOG(INFO) << "unmapped:" << std::endl;
      for (auto r : s->rays)
        LOG(INFO) << r->transpose() << std::endl;

      if (!secDrawn) {
        for (int i = 0; i < 3; ++i) {
          drawCurvedInternal(cam, *s->rays[i], *s->rays[(i + 1) % 3], img,
                             CV_BLACK, 2);
        }
      }
    }
    if (!secDrawn)
      secDrawn = true;
  }
}

void SphericalTriangulation::fillUncovered(cv::Mat &img, CameraModel *cam,
                                           cv::Scalar fillCol) {
  const int step = 10;
  double pnt[2];
  int it = 0;
  LOG(INFO) << "start finding uncovered pixels" << std::endl;
  for (int y = 0; y < img.rows; y += step)
    for (int x = 0; x < img.cols; x += step) {
      LOG(INFO) << x << ' ' << y << ' '
                << 100 * (double((it++) * step * step) / (img.rows * img.cols))
                << "%" << std::endl;
      pnt[0] = x;
      pnt[1] = y;
      if (isInConvexDummy(cam->unmap(pnt)))
        cv::rectangle(img, cv::Point(x, y), cv::Point(x + step, y + step),
                      fillCol, cv::FILLED);
    }
}

bool SphericalTriangulation::isInConvexDummy(Vec3 ray) {
  Vec3 *secRays[3];
  for (auto &r0 : _rays)
    for (auto &r1 : _rays)
      for (auto &r2 : _rays) {
        secRays[0] = &r0;
        secRays[1] = &r1;
        secRays[2] = &r2;
        if (isInSector(ray, secRays))
          return true;
      }
  return false;
}

cv::Mat SphericalTriangulation::drawTangentTri(int imWidth, int imHeight,
                                               cv::Scalar bgCol,
                                               cv::Scalar edgeCol) {
  return tangentTriang.draw(imWidth, imHeight, bgCol, edgeCol);
}

void SphericalTriangulation::draw(cv::Mat &img, CameraModel *cam,
                                  cv::Scalar edgeCol) const {

  //  Vec3 rayFrom = cam->unmap(Vec2(90.1058, 539.013).data());
  //  Vec3 rayTo = cam->unmap(Vec2(1606.07, 292.959).data());
  //  drawCurvedInternal(cam, rayFrom, rayTo, img, CV_BLACK);

  std::set<std::pair<Vec3 *, Vec3 *>> edgesDrawn;
  for (auto triSec : _sectors) {
    for (int i = 0; i < 3; ++i) {
      Vec3 *rayFromPtr = triSec.second.rays[i];
      Vec3 *rayToPtr = triSec.second.rays[(i + 1) % 3];
      if (rayFromPtr > rayToPtr)
        std::swap(rayFromPtr, rayToPtr);
      if (edgesDrawn.find({rayFromPtr, rayToPtr}) != edgesDrawn.end())
        continue;
      edgesDrawn.insert({rayFromPtr, rayToPtr});
      drawCurvedInternal(cam, *rayFromPtr, *rayToPtr, img, edgeCol, 1);
    }
  }
}

} // namespace mdso
