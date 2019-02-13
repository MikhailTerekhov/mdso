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
    DelaunayDsoInitializer::DebugOutputType debugOutputType)
    : cam(cam), dsoSystem(dsoSystem), pixelSelector(pixelSelector),
      stereoMatcher(cam), hasFirstFrame(false), framesSkipped(0),
      debugOutputType(debugOutputType) {}

bool DelaunayDsoInitializer::addFrame(const cv::Mat &frame,
                                      int globalFrameNum) {
  if (!hasFirstFrame) {
    frames[0] = frame;
    globalFrameNums[0] = globalFrameNum;
    hasFirstFrame = true;
    return false;
  } else {
    if (framesSkipped < FLAGS_first_frames_skip) {
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
  SE3 motion = stereoMatcher.match(frames, keyPoints, depths);

  StdVector<std::pair<Vec2, double>> lastKeyPointDepths;
  lastKeyPointDepths.reserve(keyPoints[1].size());
  for (int i = 0; i < keyPoints[1].size(); ++i)
    lastKeyPointDepths.push_back({keyPoints[1][i], depths[1][i]});

  if (dsoSystem)
    dsoSystem->lastKeyPointDepths = std::move(lastKeyPointDepths);

  return createKeyFramesDelaunay(cam, frames, globalFrameNums, keyPoints,
                                 depths, motion, pixelSelector,
                                 debugOutputType);
}

std::vector<KeyFrame> DelaunayDsoInitializer::createKeyFramesDelaunay(
    CameraModel *cam, cv::Mat frames[2], int frameNums[2],
    StdVector<Vec2> initialPoints[2], std::vector<double> initialDepths[2],
    const SE3 &firstToSecond, PixelSelector *pixelSelector,
    DebugOutputType debugOutputType) {
  std::vector<KeyFrame> keyFrames;
  for (int i = 0; i < 2; ++i) {
    keyFrames.push_back(KeyFrame(cam, frames[i], frameNums[i], *pixelSelector));
    for (const auto &ip : keyFrames.back().immaturePoints)
      ip->stddev = 1;
  }

  keyFrames[0].preKeyFrame->worldToThis = SE3();
  keyFrames[1].preKeyFrame->worldToThis = firstToSecond;

  if (settingUsePlainTriangulation) {
    Terrain kpTerrains[2] = {Terrain(cam, initialPoints[0], initialDepths[0]),
                             Terrain(cam, initialPoints[1], initialDepths[1])};
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
      depthedRays[kfInd].reserve(initialPoints[kfInd].size());
      for (int i = 0; i < int(initialPoints[kfInd].size()); ++i)
        depthedRays[kfInd].push_back(
            cam->unmap(initialPoints[kfInd][i].data()).normalized() *
            initialDepths[kfInd][i]);
    }

    SphericalTerrain kpTerrains[2] = {SphericalTerrain(depthedRays[0]),
                                      SphericalTerrain(depthedRays[1])};

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
        int pointsNeeded = settingInterestPointsUsed *
                           (static_cast<double>(pointsTotal) / pointsInTriang);
        if (i != reselectCount) {
          keyFrames[kfInd].selectPointsDenser(*pixelSelector, pointsNeeded);
        }
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
        insertDepths(img, initialPoints[1], initialDepths[1], minDepthCol,
                     maxDepthCol, true);

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
