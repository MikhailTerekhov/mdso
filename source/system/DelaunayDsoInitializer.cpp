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
    const Settings::DelaunayDsoInitializer &initSettings,
    const Settings::StereoMatcher &smSettings,
    const Settings::Threading &threadingSettings,
    const Settings::Triangulation &triangulationSettings,
    const Settings::KeyFrame &kfSettings,
    const Settings::PointTracer &tracingSettings,
    const Settings::Intencity &intencitySettings,
    const Settings::ResidualPattern &rpSettings,
    const Settings::Pyramid &pyrSettings)
    : cam(cam)
    , dsoSystem(dsoSystem)
    , pixelSelector(pixelSelector)
    , stereoMatcher(cam, smSettings, threadingSettings)
    , hasFirstFrame(false)
    , framesSkipped(0)
    , pointsNeeded(pointsNeeded)
    , debugOutputType(debugOutputType)
    , initSettings(initSettings)
    , threadingSettings(threadingSettings)
    , triangulationSettings(triangulationSettings)
    , kfSettings(kfSettings)
    , tracingSettings(tracingSettings)
    , intencitySettings(intencitySettings)
    , rpSettings(rpSettings)
    , pyrSettings(pyrSettings) {}

bool DelaunayDsoInitializer::addFrame(const cv::Mat &frame,
                                      int globalFrameNum) {
  if (!hasFirstFrame) {
    frames[0] = frame;
    globalFrameNums[0] = globalFrameNum;
    hasFirstFrame = true;
    return false;
  } else {
    if (framesSkipped < initSettings.firstFramesSkip) {
      ++framesSkipped;
      return false;
    }

    frames[1] = frame;
    globalFrameNums[1] = globalFrameNum;
    return true;
  }
}

std::vector<KeyFrame> DelaunayDsoInitializer::createKeyFrames() {
  StdVector<Vec2> keyPoints[2];
  std::vector<double> depths[2];
  SE3 firstToSecond = stereoMatcher.match(frames, keyPoints, depths);

  StdVector<std::pair<Vec2, double>> lastKeyPointDepths;
  lastKeyPointDepths.reserve(keyPoints[1].size());
  for (int i = 0; i < keyPoints[1].size(); ++i)
    lastKeyPointDepths.push_back({keyPoints[1][i], depths[1][i]});

  if (dsoSystem)
    dsoSystem->lastKeyPointDepths = std::move(lastKeyPointDepths);

  std::vector<KeyFrame> keyFrames;
  for (int i = 0; i < 2; ++i) {
    keyFrames.push_back(KeyFrame(cam, frames[i], globalFrameNums[i],
                                 *pixelSelector, kfSettings, tracingSettings,
                                 intencitySettings, rpSettings, pyrSettings));
    for (const auto &ip : keyFrames.back().immaturePoints)
      ip->stddev = 1;
  }

  keyFrames[0].preKeyFrame->worldToThis = SE3();
  keyFrames[1].preKeyFrame->worldToThis = firstToSecond;

  if (initSettings.usePlainTriangulation) {
    Terrain kpTerrains[2] = {
        Terrain(cam, keyPoints[0], depths[0], triangulationSettings),
        Terrain(cam, keyPoints[1], depths[1], triangulationSettings)};
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
        SphericalTerrain(depthedRays[0], triangulationSettings),
        SphericalTerrain(depthedRays[1], triangulationSettings)};

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

    std::vector<double> ipDepths;
    ipDepths.reserve(keyFrames[1].immaturePoints.size());
    for (const auto &ip : keyFrames[1].immaturePoints)
      ipDepths.push_back(ip->depth);
    setDepthColBounds(ipDepths);

    if (debugOutputType != NO_DEBUG) {
      if (FLAGS_show_interpolation || FLAGS_write_files) {
        cv::Mat img = keyFrames[1].preKeyFrame->frameColored.clone();
        insertDepths(img, keyPoints[1], depths[1], minDepthCol, maxDepthCol,
                     true);

        // cv::circle(img, cv::Point(1268, 173), 7, CV_BLACK, 2);

        auto &ips = keyFrames[1].immaturePoints;
        StdVector<Vec2> pnts;
        pnts.reserve(ips.size());
        std::vector<double> d;
        d.reserve(ips.size());
        for (const auto &ip : ips) {
          pnts.push_back(ip->p);
          d.push_back(ip->depth);
        }

        kpTerrains[1].draw(img, cam, CV_GREEN, minDepthCol, maxDepthCol);
        insertDepths(img, pnts, d, minDepthCol, maxDepthCol, false);

        if (FLAGS_show_interpolation) {
          cv::imshow("interpolated", img);
          cv::waitKey();
        }

        if (FLAGS_write_files)
          cv::imwrite(FLAGS_output_directory + "/interpolated.jpg", img);
      }
    }
  }

  return keyFrames;
}

} // namespace fishdso
