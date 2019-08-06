#include "util/DistanceMap.h"
#include "util/defs.h"
#include "util/settings.h"
#include <glog/logging.h>
#include <queue>

namespace fishdso {

// lowest power of 2 greater than x
unsigned clp2(unsigned x) {
  x--;
  x |= (x >> 1);
  x |= (x >> 2);
  x |= (x >> 4);
  x |= (x >> 8);
  x |= (x >> 16);
  return x + 1;
}

DistanceMap::MapEntry::MapEntry(int givenW, int givenH,
                                const StdVector<Vec2> &points,
                                const Settings::DistanceMap &settings) {
  const int wdiv = 1 + (givenW - 1) / settings.maxWidth;
  const int hdiv = 1 + (givenH - 1) / settings.maxHeight;
  pyrDown = std::min(clp2(wdiv), clp2(hdiv));
  const int w = givenW / pyrDown, h = givenH / pyrDown;

  LOG(INFO) << "DistanceMap: w, h = " << w << ' ' << h
            << " pyrDown = " << pyrDown << std::endl;

  dist = MatXXi::Constant(h, w, std::numeric_limits<int>::max());

  StdQueue<Vec2i> q;

  for (const Vec2 &p : points) {
    if (!(p[0] >= 0 && p[0] < givenW && p[1] >= 0 && p[1] < givenH)) {
      continue;
    }
    Vec2i pi = p.cast<int>() / pyrDown;
    dist(pi[1], pi[0]) = 0;
    q.push(pi);
  }

  while (!q.empty()) {
    Vec2i cur = q.front();
    int curd = dist(cur[1], cur[0]);
    q.pop();

    for (int dx : {-1, 1}) {
      const int newx = cur[0] + dx;
      if (newx >= 0 && newx < w && dist(cur[1], newx) > curd + 1) {
        dist(cur[1], newx) = curd + 1;
        q.push(Vec2i(newx, cur[1]));
      }
    }
    for (int dy : {-1, 1}) {
      const int newy = cur[1] + dy;
      if (newy >= 0 && newy < h && dist(newy, cur[0]) > curd + 1) {
        dist(newy, cur[0]) = curd + 1;
        q.push(Vec2i(cur[0], newy));
      }
    }
  }
}

DistanceMap::DistanceMap(CameraBundle *cam, StdVector<Vec2> points[],
                         const Settings::DistanceMap &settings)
    : camCount(cam->bundle.size())
    , settings(settings) {
  for (int i = 0; i < camCount; ++i)
    maps.emplace_back(cam->bundle[i].cam.getWidth(),
                      cam->bundle[i].cam.getHeight(), points[i], settings);
}

struct PointDist {
  int cam, ind, dist;
};

int DistanceMap::choose(StdVector<Vec2> otherPoints[], int pointsNeeded,
                        std::vector<int> chosenIndices[]) {
  int totalPoints = std::accumulate(
      otherPoints, otherPoints + camCount, 0,
      [](int size, const auto &pointsVec) { return size + pointsVec.size(); });
  std::vector<PointDist> otherDist;
  otherDist.reserve(totalPoints);
  for (int ci = 0; ci < camCount; ++ci)
    for (int pi = 0; pi < otherPoints[ci].size(); ++pi) {
      Vec2i pnt = otherPoints[ci][pi].cast<int>() / maps[ci].pyrDown;
      int curDist =
          Eigen::AlignedBox2i(Vec2i::Zero(), Vec2i(maps[ci].dist.cols() - 1,
                                                   maps[ci].dist.rows() - 1))
                  .contains(pnt)
              ? maps[ci].dist(pnt[1], pnt[0])
              : -1;
      otherDist.push_back({ci, pi, curDist});
    }

  int total = std::min(pointsNeeded, int(otherDist.size()));
  int result = 0;
  std::nth_element(
      otherDist.begin(), otherDist.begin() + total, otherDist.end(),
      [](const auto &a, const auto &b) { return a.dist > b.dist; });
  for (int i = 0; i < total; ++i) {
    const PointDist &p = otherDist[i];
    if (p.dist != -1) {
      chosenIndices[p.cam].push_back(p.ind);
      ++result;
    }
  }

  return result;
}

} // namespace fishdso
