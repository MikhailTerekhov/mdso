#include "output/InterpolationDrawer.h"
#include "util/util.h"

namespace fishdso {

InterpolationDrawer::InterpolationDrawer(CameraModel *cam)
    : cam(cam)
    , mDidInitialize(false) {}

void InterpolationDrawer::initialized(
    const KeyFrame *lastKeyFrame, const SphericalTerrain *lastTerrain,
    const StdVector<Vec2> &keyPoints,
    const std::vector<double> &keyPointDepths) {
  std::vector<double> ipDepths;
  ipDepths.reserve(lastKeyFrame->immaturePoints.size());
  for (const auto &ip : lastKeyFrame->immaturePoints)
    ipDepths.push_back(ip->depth);
  setDepthColBounds(ipDepths);

  result = lastKeyFrame->preKeyFrame->frameColored.clone();
  insertDepths(result, keyPoints, keyPointDepths, minDepthCol, maxDepthCol,
               true);

  // cv::circle(img, cv::Point(1268, 173), 7, CV_BLACK, 2);

  auto &ips = lastKeyFrame->immaturePoints;
  StdVector<Vec2> pnts;
  pnts.reserve(ips.size());
  std::vector<double> d;
  d.reserve(ips.size());
  for (const auto &ip : ips) {
    pnts.push_back(ip->p);
    d.push_back(ip->depth);
  }

  lastTerrain->draw(result, cam, CV_GREEN, minDepthCol, maxDepthCol);
  insertDepths(result, pnts, d, minDepthCol, maxDepthCol, false);

  mDidInitialize = true;
}

bool InterpolationDrawer::didInitialize() { return mDidInitialize; }

cv::Mat3b InterpolationDrawer::draw() { return result; }

} // namespace fishdso
