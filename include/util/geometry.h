#include "util/types.h"

namespace fishdso {

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
bool isInSector(const Vec3 &ray, Vec3 *s[3]);

} // namespace fishdso
