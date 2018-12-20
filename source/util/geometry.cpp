#include "util/geometry.h"
#include <glog/logging.h>
#include <cmath>

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
  return !(isInsideTriangle(b, c, d, a) || isInsideTriangle(a, c, d, b) ||
           isInsideTriangle(a, b, d, c) || isInsideTriangle(a, b, c, d));
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

bool isInSector(const Vec3 &ray, Vec3 *s[3]) {
  static const Vec3 t(0.0, 0.0, 1.0);

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
  if (angle1 > sectorAngle && angle2 > sectorAngle)
    return false;
  if (angle1 < sectorAngle && angle2 < sectorAngle)
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
