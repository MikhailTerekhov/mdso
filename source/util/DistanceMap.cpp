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

DistanceMap::DistanceMap(int givenW, int givenH,
                         const StdVector<Vec2> &points) {
  const int wdiv = 1 + (givenW - 1) / settingMaxDistMapW;
  const int hdiv = 1 + (givenH - 1) / settingMaxDistMapH;
  pyrDown = std::min(clp2(wdiv), clp2(hdiv));
  const int w = givenW / pyrDown, h = givenH / pyrDown;

  std::cout << "w, h = " << w << ' ' << h << "\npyrDown = " << pyrDown
            << std::endl;

  dist = MatXXi::Constant(h, w, std::numeric_limits<int>::max());

  StdQueue<Vec2i> q;

  for (const Vec2 &p : points) {
    LOG_IF(WARNING, !(p[0] >= 0 && p[0] < givenW && p[1] >= 0 && p[1] < givenH))
        << "point in DistanceMap::DistanceMap OOB! w, h = " << givenW << ' '
        << givenH << " p = " << p.transpose() << std::endl;
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

std::vector<bool> DistanceMap::choose(const StdVector<Vec2> &otherPoints,
                                      int pointsNeeded) {
  std::vector<bool> used(otherPoints.size(), false);
  std::vector<std::pair<int, int>> otherDist(otherPoints.size());
  for (int i = 0; i < otherPoints.size(); ++i) {
    const Vec2 &p = otherPoints[i];
    Vec2i pi = otherPoints[i].cast<int>() / pyrDown;
    if (!(pi[0] >= 0 && pi[0] < dist.cols() && pi[1] >= 0 && pi[1] < dist.rows())) {
      LOG(WARNING) << "point in DistanceMap::choose OOB! w, h = " << dist.cols()
                   << ' ' << dist.rows() << " p = " << pi.transpose()
                   << std::endl;
      otherDist[i].first = i;
      otherDist[i].second = -1;
    } else {
      otherDist[i].first = i;
      otherDist[i].second = dist(pi[1], pi[0]);
    }
  }
  std::sort(otherDist.begin(), otherDist.end(),
            [](const auto &a, const auto &b) { return a.second > b.second; });
  int total = std::min(pointsNeeded, int(otherPoints.size()));
  for (int i = 0; i < total; ++i)
    if (otherDist[i].second != -1)
      used[otherDist[i].first] = true;

  return used;
}

} // namespace fishdso
