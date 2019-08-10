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

DistanceMap::DistanceMap(int givenW, int givenH, const StdVector<Vec2> &points,
                         const Settings::DistanceMap &settings)
    : settings(settings) {
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

std::vector<int> DistanceMap::choose(const StdVector<Vec2> &otherPoints,
                                     int pointsNeeded) {
  std::vector<int> chosen;
  std::vector<std::pair<int, int>> otherDist(otherPoints.size());
  for (int i = 0; i < otherPoints.size(); ++i) {
    const Vec2 &p = otherPoints[i];
    Vec2i pi = p.cast<int>() / pyrDown;

    otherDist[i].first = i;
    if (!Eigen::AlignedBox2i(Vec2i::Zero(), Vec2i(dist.cols() - 1, dist.rows() - 1))
             .contains(pi))
      otherDist[i].second = -1;
    else
      otherDist[i].second = dist(pi[1], pi[0]);
  }

  int total = std::min(pointsNeeded, int(otherPoints.size()));
  std::nth_element(
      otherDist.begin(), otherDist.begin() + total, otherDist.end(),
      [](const auto &a, const auto &b) { return a.second > b.second; });
  for (int i = 0; i < total; ++i)
    if (otherDist[i].second != -1)
      chosen.push_back(otherDist[i].first);

  return chosen;
}

} // namespace fishdso
