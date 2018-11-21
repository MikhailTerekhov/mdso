#include "util/geometry.h"

namespace fishdso {

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

bool isSameSide(const Vec2 &a, const Vec2 &b,
                                    const Vec2 &p1, const Vec2 &p2) {
  return cross2(b - a, p1 - a) * cross2(b - a, p2 - a) >= 0;
}

bool isInsideTriangle(const Vec2 &a, const Vec2 &b,
                                          const Vec2 &c, const Vec2 &p) {
  return isSameSide(a, b, p, c) && isSameSide(a, c, p, b) &&
         isSameSide(b, c, p, a);
}

bool isABCDConvex(const Vec2 &a, const Vec2 &b,
                                      const Vec2 &c, const Vec2 &d) {
  return !(isInsideTriangle(b, c, d, a) || isInsideTriangle(a, c, d, b) ||
           isInsideTriangle(a, b, d, c) || isInsideTriangle(a, b, c, d));
}

bool areEqual(const Vec2 &a, const Vec2 &b, double eps) {
  return (a - b).norm() < eps;
}

bool doesABcontain(const Vec2 &a, const Vec2 &b,
                                       const Vec2 &p, double eps) {
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


} // namespace fishdso
