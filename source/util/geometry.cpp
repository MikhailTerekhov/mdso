#include "util/geometry.h"
#include "util/settings.h"
#include <cmath>
#include <glog/logging.h>

namespace fishdso {

double angle(const Vec3 &a, const Vec3 &b) {
  double cosAngle = a.normalized().dot(b.normalized());
  if (cosAngle < -1)
    cosAngle = -1;
  else if (cosAngle > 1)
    cosAngle = 1;
  return std::acos(cosAngle);
}

Mat33 toEssential(const SE3 &motion) {
  return SO3::hat(motion.translation()) * motion.rotationMatrix();
}

Vec2 triangulate(const SE3 &firstToSecond, const Vec3 &firstRay,
                 const Vec3 &secondRay) {
  Mat32 A;
  A.col(0) = firstToSecond.so3() * firstRay;
  A.col(1) = -secondRay;

  return A.fullPivHouseholderQr().solve(-firstToSecond.translation());
}

double cross2(const Vec2 &a, const Vec2 &b) {
  return a[0] * b[1] - a[1] * b[0];
}

bool isSameSide(const Vec2 &a, const Vec2 &b, const Vec2 &p1, const Vec2 &p2) {
  return cross2(b - a, p1 - a) * cross2(b - a, p2 - a) >= 0;
}

bool isInsideTriangle(const Vec2 &a, const Vec2 &b, const Vec2 &c,
                      const Vec2 &p) {
  return isSameSide(a, b, p, c) && isSameSide(a, c, p, b) &&
         isSameSide(b, c, p, a);
}

bool isABCDConvex(const Vec2 &a, const Vec2 &b, const Vec2 &c, const Vec2 &d) {
  return isSameSide(a, b, c, d) && isSameSide(b, c, a, d) &&
         isSameSide(c, d, a, b) && isSameSide(d, a, b, c);
}

bool areEqual(const Vec2 &a, const Vec2 &b, double eps) {
  return (a - b).norm() < eps;
}

bool doesABcontain(const Vec2 &a, const Vec2 &b, const Vec2 &p, double eps) {
  Vec2 ap = p - a;
  double abNorm = (b - a).norm();
  Vec2 abN = (b - a) / abNorm;
  double abNp = abN.dot(ap);
  return std::abs(ap.dot(Vec2(abN[1], -abN[0]))) < eps && abNp >= -eps &&
         abNp <= abNorm + eps;
}

bool isABDelaunay(const Vec2 &a, const Vec2 &b, const Vec2 &c, const Vec2 &d) {
  Vec2 da = a - d;
  Vec2 db = b - d;
  Vec2 dc = c - d;
  Vec2 ab = b - a;
  Vec2 bc = c - b;
  Mat33 checker;
  // clang-format off
  checker << da[0], da[1], da.squaredNorm(),
             db[0], db[1], db.squaredNorm(),
             dc[0], dc[1], dc.squaredNorm();
  // clang-format on

  return checker.determinant() * cross2(ab, bc) <= 0;
}

bool doesABIntersectCD(const Vec2 &a, const Vec2 &b, const Vec2 &c,
                       const Vec2 &d) {
  const double eps = settingEpsPointIsOnSegment;
  if (!(((a[0] + eps < c[0] && c[0] < b[0] - eps) ||
         (a[0] + eps < d[0] && d[0] < b[0] - eps)) &&
        (((a[1] + eps < c[1] && c[1] < b[1] - eps) ||
          (a[1] + eps < d[1] && d[1] < b[1] - eps)))))
    return false;

  return !isSameSide(a, b, c, d) && !isSameSide(c, d, a, b);
}

// approximates a point with a given coordinate along the curve given points on
// the curve with integer coordinates
Vec2 approxOnCurve(const StdVector<Vec2> &points, double displ) {
  CHECK(points.size() >= 2);
  if (displ <= 0)
    return points[0] - displ * (points[0] - points[1]);
  if (displ + 1 >= points.size()) {
    const Vec2 &last = points.back();
    const Vec2 &lbo = points[points.size() - 2];
    double res = displ - points.size() + 1;
    return last + res * (last - lbo);
  }
  int intDispl = displ;
  double fracDispl = displ - intDispl;
  return points[intDispl] +
         fracDispl * (points[intDispl + 1] - points[intDispl]);
}

bool isInSector(const Vec3 &ray, Vec3 *s[3]) {
  Mat33 A;
  for (int i = 0; i < 3; ++i)
    A.col(i) = *s[i];
  auto Qr = A.fullPivHouseholderQr();
  Vec3 coeffsR = Qr.solve(ray);

  return coeffsR[0] >= 0 && coeffsR[1] >= 0 && coeffsR[2] >= 0;
}

bool intersectOnSphere(double sectorAngle, Vec3 &dir1, Vec3 &dir2) {
  double angle1 = angle(dir1, Vec3(0., 0., 1.));
  double angle2 = angle(dir2, Vec3(0., 0., 1.));
  static const double epsAngle = 1e-4;
  if (angle1 > sectorAngle + epsAngle && angle2 > sectorAngle + epsAngle)
    return false;
  if (angle1 <= sectorAngle + epsAngle && angle2 <= sectorAngle + epsAngle)
    return true;

  double z0 = std::cos(sectorAngle);
  double R = std::sin(sectorAngle);
  Vec3 norm = dir1.cross(dir2);
  double a = norm[0], b = norm[1], c = norm[2];
  double a2b2 = a * a + b * b;
  double x0 = -(a * c * z0) / a2b2, y0 = -(b * c * z0) / a2b2;
  double u = a * y0 - b * x0, v = x0 * x0 + y0 * y0 - R * R;
  double sD4 = std::sqrt(u * u - a2b2 * v);
  double t1 = (u - sD4) / a2b2, t2 = (u + sD4) / a2b2;
  Vec3 sol1(x0 - t1 * b, y0 + t1 * a, z0), sol2(x0 - t2 * b, y0 + t2 * a, z0);
  double sol1Norm2 = sol1.squaredNorm(), sol2Norm2 = sol2.squaredNorm();
  CHECK(std::abs(sol1Norm2 - 1.0) < 1e-4)
      << "forceCamValidity failed! |sol1| = " << std::sqrt(sol1Norm2)
      << std::endl;
  CHECK(std::abs(sol2Norm2 - 1.0) < 1e-4)
      << "forceCamValidity failed! |sol2| = " << std::sqrt(sol2Norm2)
      << std::endl;
  Mat32 M;
  M << dir1, dir2;
  Vec2 coeffs = M.fullPivHouseholderQr().solve(sol1);
  Vec3 &sol = (coeffs[0] >= 0 && coeffs[1] >= 0) ? sol1 : sol2;

  if (angle1 > sectorAngle)
    dir1 = sol;
  else if (angle2 > sectorAngle)
    dir2 = sol;

  return true;
}

} // namespace fishdso
