#include "system/DelaunayDsoInitializer.h"
#include "util/SphericalTerrain.h"
#include "util/defs.h"
#include "util/util.h"
#include <algorithm>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>

namespace mdso {

DelaunayDsoInitializer::DelaunayDsoInitializer(
    DsoSystem *dsoSystem, CameraBundle *cam, PixelSelector pixelSelectors[],
    const std::vector<DelaunayInitializerObserver *> &observers,
    const InitializerSettings &_settings)
    : cam(cam)
    , dsoSystem(dsoSystem)
    , pixelSelectors(pixelSelectors)
    , hasFirstFrame(false)
    , framesSkipped(0)
    , settings(_settings)
    , observers(observers) {}

void DelaunayDsoInitializer::setImages() {
  for (int i = 0; i < 2; ++i) {
    framesGray[i] = cvtBgrToGray(frames[i]);
    cv::Mat1d gradX, gradY;
    grad(framesGray[i], gradX, gradY, gradNorm[i]);
  }
}

bool DelaunayDsoInitializer::addMultiFrame(const cv::Mat newFrames[],
                                           Timestamp newTimestamps[]) {
  if (!hasFirstFrame) {
    frames[0] = newFrames[0];
    timestamps[0] = newTimestamps[0];
    hasFirstFrame = true;
    return false;
  } else {
    if (framesSkipped < settings.initializer.firstFramesSkip) {
      ++framesSkipped;
      return false;
    }

    frames[1] = newFrames[0];
    timestamps[1] = newTimestamps[0];

    setImages();

    return true;
  }
}

DsoInitializer::InitializedVector DelaunayDsoInitializer::initialize() {
  CHECK(cam->bundle.size() == 1) << "Multicamera case is NIY";

  StdVector<Vec2> keyPoints[2];
  std::vector<double> depths[2];

  StereoMatcher stereoMatcher(&cam->bundle[0].cam, settings.stereoMatcher,
                              settings.threading);

  SE3 firstToSecond = stereoMatcher.match(frames, keyPoints, depths);

  InitializedVector initFrames;
  for (int i = 0; i < 2; ++i)
    initFrames.emplace_back(&frames[i], &timestamps[i], 1);

  initFrames[0].thisToWorld = SE3();
  initFrames[1].thisToWorld = cam->bundle[0].thisToBody *
                              firstToSecond.inverse() *
                              cam->bundle[0].bodyToThis;

  if (settings.initializer.usePlainTriangulation) {
    Terrain terrains[2] = {Terrain(&cam->bundle[0].cam, keyPoints[0], depths[0],
                                   settings.triangulation),
                           Terrain(&cam->bundle[0].cam, keyPoints[1], depths[1],
                                   settings.triangulation)};
    for (int i = 0; i < 2; ++i) {
      PixelSelector::PointVector points = pixelSelectors[i].select(
          frames[i], gradNorm[i], settings.keyFrame.immaturePointsNum());
      initFrames[i].frames[0].depthedPoints.reserve(points.size());
      for (const cv::Point &cvp : points) {
        Vec2 p = toVec2(cvp);
        double depth = 0;
        if (terrains[i](p, depth))
          initFrames[i].frames[0].depthedPoints.push_back({p, depth});
      }
    }
  } else {
    std::vector<Vec3> depthedRays[2];
    for (int kfInd = 0; kfInd < 2; ++kfInd) {
      depthedRays[kfInd].reserve(keyPoints[kfInd].size());
      for (int i = 0; i < int(keyPoints[kfInd].size()); ++i)
        depthedRays[kfInd].push_back(
            cam->bundle[0].cam.unmap(keyPoints[kfInd][i].data()).normalized() *
            depths[kfInd][i]);
    }

    SphericalTerrain kpTerrains[2] = {
        SphericalTerrain(depthedRays[0], settings.triangulation),
        SphericalTerrain(depthedRays[1], settings.triangulation)};

    for (int kfInd = 0; kfInd < 2; ++kfInd) {
      const int reselectCount = 1;
      int pointsNeeded = settings.keyFrame.immaturePointsNum();
      for (int ir = 0; ir < reselectCount + 1; ++ir) {
        PixelSelector::PointVector points =
            pixelSelectors[0].select(frames[kfInd], gradNorm[kfInd],
                                     settings.keyFrame.immaturePointsNum());
        for (const cv::Point &cvp : points) {
          Vec2 p = toVec2(cvp);
          double depth;
          if (kpTerrains[kfInd](cam->bundle[0].cam.unmap(p), depth))
            initFrames[kfInd].frames[0].depthedPoints.push_back({p, depth});
        }

        int pointsTotal = points.size();
        int pointsInTriang = initFrames[kfInd].frames[0].depthedPoints.size();
        pointsNeeded *= (double(pointsTotal) / pointsInTriang);
      }
    }

    Vec2 *keyPointsData[] = {keyPoints[0].data(), keyPoints[1].data()};
    double *depthsData[] = {depths[0].data(), depths[1].data()};
    int sizes[] = {int(keyPoints[0].size()), int(keyPoints[1].size())};
    for (DelaunayInitializerObserver *obs : observers)
      obs->initialized(initFrames.data(), kpTerrains, keyPointsData, depthsData,
                       sizes);
  }

  return initFrames;
}

} // namespace mdso
