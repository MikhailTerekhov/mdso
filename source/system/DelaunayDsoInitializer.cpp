#include "system/DelaunayDsoInitializer.h"
#include "util/SphericalTerrain.h"
#include "util/defs.h"
#include "util/util.h"
#include <algorithm>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>

namespace fishdso {

DelaunayDsoInitializer::DelaunayDsoInitializer(
    DsoSystem *dsoSystem, CameraModel *cam, PixelSelector *pixelSelector,
    int pointsNeeded, DelaunayDsoInitializer::DebugOutputType debugOutputType,
    const std::vector<InitializerObserver *> &observers,
    const InitializerSettings &_settings)
    : cam(cam)
    , dsoSystem(dsoSystem)
    , pixelSelector(pixelSelector)
    , stereoMatcher(cam, _settings.stereoMatcher, _settings.threading)
    , hasFirstFrame(false)
    , framesSkipped(0)
    , pointsNeeded(pointsNeeded)
    , debugOutputType(debugOutputType)
    , settings(_settings)
    , observers(observers) {}

bool DelaunayDsoInitializer::addFrame(const cv::Mat &frame,
                                      int globalFrameNum) {
  if (!hasFirstFrame) {
    frames[0] = frame;
    globalFrameNums[0] = globalFrameNum;
    hasFirstFrame = true;
    return false;
  } else {
    if (framesSkipped < settings.initializer.firstFramesSkip) {
      ++framesSkipped;
      return false;
    }

    frames[1] = frame;
    globalFrameNums[1] = globalFrameNum;
    return true;
  }
}

StdVector<KeyFrame> DelaunayDsoInitializer::createKeyFrames() {
  StdVector<Vec2> keyPoints[2];
  std::vector<double> depths[2];
  SE3 firstToSecond = stereoMatcher.match(frames, keyPoints, depths);

  StdVector<std::pair<Vec2, double>> lastKeyPointDepths;
  lastKeyPointDepths.reserve(keyPoints[1].size());
  for (int i = 0; i < keyPoints[1].size(); ++i)
    lastKeyPointDepths.push_back({keyPoints[1][i], depths[1][i]});

  if (dsoSystem)
    dsoSystem->lastKeyPointDepths = std::move(lastKeyPointDepths);

  StdVector<KeyFrame> keyFrames;
  for (int i = 0; i < 2; ++i) {
    keyFrames.push_back(KeyFrame(cam, frames[i], globalFrameNums[i],
                                 *pixelSelector, settings.keyFrame,
                                 settings.tracingSettings));
    for (const auto &ip : keyFrames.back().immaturePoints)
      ip->stddev = 1;
  }

  keyFrames[0].thisToWorld = SE3();
  keyFrames[1].thisToWorld = firstToSecond.inverse();

  if (settings.initializer.usePlainTriangulation) {
    Terrain kpTerrains[2] = {
        Terrain(cam, keyPoints[0], depths[0], settings.triangulation),
        Terrain(cam, keyPoints[1], depths[1], settings.triangulation)};
    for (int i = 0; i < 2; ++i) {
      for (const auto &ip : keyFrames[i].immaturePoints) {
        double depth;
        if (kpTerrains[i](ip->p, depth)) {
          ip->state = ImmaturePoint::ACTIVE;
          ip->depth = depth;
        } else
          ip->state = ImmaturePoint::OOB;
      }
    }
  } else {
    std::vector<Vec3> depthedRays[2];
    for (int kfInd = 0; kfInd < 2; ++kfInd) {
      depthedRays[kfInd].reserve(keyPoints[kfInd].size());
      for (int i = 0; i < int(keyPoints[kfInd].size()); ++i)
        depthedRays[kfInd].push_back(
            cam->unmap(keyPoints[kfInd][i].data()).normalized() *
            depths[kfInd][i]);
    }

    SphericalTerrain kpTerrains[2] = {
        SphericalTerrain(depthedRays[0], settings.triangulation),
        SphericalTerrain(depthedRays[1], settings.triangulation)};

    for (int kfInd = 0; kfInd < 2; ++kfInd) {
      const int reselectCount = 1;
      for (int i = 0; i < reselectCount + 1; ++i) {
        for (const auto &ip : keyFrames[kfInd].immaturePoints) {
          double depth;
          if (ip->state == ImmaturePoint::ACTIVE &&
              kpTerrains[kfInd](cam->unmap(ip->p.data()), depth)) {
            ip->state = ImmaturePoint::ACTIVE;
            ip->depth = depth;
          } else
            ip->state = ImmaturePoint::OOB;
        }

        int pointsTotal = keyFrames[kfInd].immaturePoints.size();

        auto it = keyFrames[kfInd].immaturePoints.begin();
        while (it != keyFrames[kfInd].immaturePoints.end()) {
          if ((*it)->state != ImmaturePoint::ACTIVE)
            it = keyFrames[kfInd].immaturePoints.erase(it);
          else
            it++;
        }

        int pointsInTriang = keyFrames[kfInd].immaturePoints.size();
        int newPointsNeeded =
            pointsNeeded * (static_cast<double>(pointsTotal) / pointsInTriang);
        if (i != reselectCount)
          keyFrames[kfInd].selectPointsDenser(*pixelSelector, newPointsNeeded);
      }
    }

    for (InitializerObserver *obs : observers)
      obs->initialized(&keyFrames[1], &kpTerrains[1], keyPoints[1], depths[1]);
  }

  return keyFrames;
}

} // namespace fishdso
