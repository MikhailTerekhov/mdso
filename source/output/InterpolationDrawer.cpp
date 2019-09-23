#include "output/InterpolationDrawer.h"
#include "util/util.h"

namespace fishdso {

InterpolationDrawer::InterpolationDrawer(CameraModel *cam)
    : cam(cam)
    , mDidInitialize(false) {}

void InterpolationDrawer::initialized(const InitializedFrame frames[],
                                      const SphericalTerrain terrains[],
                                      Vec2 *keyPoints[],
                                      double *keyPointDepths[], int sizes[]) {
  const InitializedFrame &lastFrame = frames[1];
  std::vector<double> initDepths;
  initDepths.reserve(lastFrame.frames[0].depthedPoints.size());
  for (const auto &[p, d] : lastFrame.frames[0].depthedPoints)
    initDepths.push_back(d);
  setDepthColBounds(initDepths);

  result = lastFrame.frames[0].frame.clone();
  insertDepths(result, keyPoints[1], keyPointDepths[1], sizes[1], minDepthCol,
               maxDepthCol, true);

  // cv::circle(img, cv::Point(1268, 173), 7, CV_BLACK, 2);

  auto &ips = lastFrame.frames[0].depthedPoints;
  StdVector<Vec2> pnts;
  pnts.reserve(ips.size());
  std::vector<double> d;
  d.reserve(ips.size());
  for (const auto &[p, depth] : ips) {
    pnts.push_back(p);
    d.push_back(depth);
  }

  terrains[1].draw(result, cam, CV_GREEN, minDepthCol, maxDepthCol);
  insertDepths(result, pnts.data(), d.data(), pnts.size(), minDepthCol,
               maxDepthCol, false);

  mDidInitialize = true;
}

bool InterpolationDrawer::didInitialize() { return mDidInitialize; }

cv::Mat3b InterpolationDrawer::draw() { return result; }

} // namespace fishdso
