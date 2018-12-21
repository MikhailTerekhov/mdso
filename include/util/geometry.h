#include "util/types.h"

namespace fishdso {

double angle(const Vec3 &a, const Vec3 &b);
Mat33 toEssential(const SE3 &motion);
Vec2 triangulate(const SE3 &firstToSecond, const Vec3 &firstRay,
                 const Vec3 &secondRay);
double cross2(const Vec2 &a, const Vec2 &b);
bool isSameSide(const Vec2 &a, const Vec2 &b, const Vec2 &p1, const Vec2 &p2);
bool isInsideTriangle(const Vec2 &a, const Vec2 &b, const Vec2 &c,
                      const Vec2 &p);
bool isABCDConvex(const Vec2 &a, const Vec2 &b, const Vec2 &c, const Vec2 &d);
bool areEqual(const Vec2 &a, const Vec2 &b, double eps);
bool doesABcontain(const Vec2 &a, const Vec2 &b, const Vec2 &p, double eps);
bool isABDelaunay(const Vec2 &a, const Vec2 &b, const Vec2 &c,
                                   const Vec2 &d);
// doesn't count intersections by segment ends
bool doesABIntersectCD(const Vec2 &a, const Vec2 &b, const Vec2 &c,
                       const Vec2 &d);

bool isInSector(const Vec3 &ray, Vec3 *s[3]);

// intersects a spherical segment [dir1, dir2] with a sector around (0, 0, 1)
// with angle sectorAngle
bool intersectOnSphere(double sectorAngle, Vec3 &dir1, Vec3 &dir2);

} // namespace fishdso
