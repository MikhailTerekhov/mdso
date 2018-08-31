#include "system/dsoinitializer.h"
#include "util/defs.h"
#include "util/sphericalterrain.h"
#include "util/util.h"
#include <algorithm>
#include <opencv2/opencv.hpp>

namespace fishdso {

DsoInitializer::DsoInitializer(CameraModel *cam)
    : cam(cam), stereoMatcher(cam), hasFirstFrame(false), framesSkipped(0) {}

bool DsoInitializer::addFrame(const cv::Mat &frame) {
  if (!hasFirstFrame) {
    addFirstFrame(frame);
    return false;
  } else {
    if (framesSkipped < settingFirstFramesSkip) {
      ++framesSkipped;
      return false;
    }
    frames[1] = frame;
    return true;
  }
}

std::vector<KeyFrame> DsoInitializer::createKeyFrames(
    DsoInitializer::DebugOutputType debugOutputType) {
  return createKeyFramesFromStereo(NORMAL, debugOutputType);
}

void DsoInitializer::addFirstFrame(const cv::Mat &frame) {
  frames[0] = frame;
  hasFirstFrame = true;
}

std::vector<KeyFrame> DsoInitializer::createKeyFramesFromStereo(
    InterpolationType interpolationType,
    DsoInitializer::DebugOutputType debugOutputType) {
  std::vector<Vec2> keyPoints[2];
  std::vector<double> depths[2];
  stereoMatcher.match(frames, keyPoints, depths);

  std::vector<KeyFrame> keyFrames;
  for (int i = 0; i < 2; ++i)
    keyFrames.push_back(KeyFrame(frames[i]));

  if (interpolationType == PLAIN) {
    Terrain kpTerrains[2] = {Terrain(cam, keyPoints[0], depths[0]),
                             Terrain(cam, keyPoints[1], depths[1])};
    for (int i = 0; i < 2; ++i)
      for (InterestPoint &ip : keyFrames[i].interestPoints) {
        double depth;
        if (kpTerrains[i](ip.p, depth))
          ip.depth = depth;
      }
  } else if (interpolationType == NORMAL) {
    std::vector<Vec3> depthedRays[2];
    for (int kfInd = 0; kfInd < 2; ++kfInd) {
      depthedRays[kfInd].reserve(keyPoints[kfInd].size());
      for (int i = 0; i < int(keyPoints[kfInd].size()); ++i)
        depthedRays[kfInd].push_back(
            cam->unmap(keyPoints[kfInd][i].data()).normalized() *
            depths[kfInd][i]);
    }

    SphericalTerrain kpTerrains[2] = {SphericalTerrain(depthedRays[0]),
                                      SphericalTerrain(depthedRays[1])};
    for (int i = 0; i < 2; ++i)
      for (InterestPoint &ip : keyFrames[i].interestPoints) {
        double depth;
        if (kpTerrains[i](cam->unmap(ip.p.data()), depth))
          ip.depth = depth;
      }

    for (int i = 0; i < 2; ++i) {
      auto it = std::remove_if(keyFrames[i].interestPoints.begin(),
                               keyFrames[i].interestPoints.end(),
                               [](InterestPoint p) { return p.depth < 0; });
      keyFrames[i].interestPoints.resize(it -
                                         keyFrames[i].interestPoints.begin());
    }

    if (debugOutputType != NO_DEBUG) {
      cv::Mat img = keyFrames[1].frameColored.clone();
      // KpTerrains[1].draw(img, CV_GREEN);

      std::vector<std::pair<Vec2, double>> keyPairs(keyPoints[1].size());
      for (int i = 0; i < int(keyPoints[1].size()); ++i)
        keyPairs[i] = {keyPoints[1][i], depths[1][i]};
      std::sort(keyPairs.begin(), keyPairs.end(),
                [](auto a, auto b) { return a.second < b.second; });

      int padding = int(0.3 * keyPairs.size());
      double minDepth = keyPairs[0].second, maxDepth = keyPairs[padding].second;

      insertDepths(img, keyPoints[1], depths[1], minDepth, maxDepth, true);

      // cv::circle(img, cv::Point(850, 250), 7, CV_BLACK, 2);

      std::vector<InterestPoint> &ip = keyFrames[1].interestPoints;
      std::vector<Vec2> pnts(ip.size());
      std::vector<double> d(ip.size());
      for (int i = 0; i < int(ip.size()); ++i) {
        pnts[i] = ip[i].p;
        d[i] = ip[i].depth;
      }

      //    drawCurvedInternal(cam, Vec2(100.0, 100.0), Vec2(1000.0, 500.0),
      //    img,
      //                       CV_BLACK);

      // KpTerrains[1].drawCurved(cam, img, CV_GREEN);

      kpTerrains[1].draw(img, cam, CV_GREEN, minDepth, maxDepth);

      if (debugOutputType == SPARSE_DEPTHS) {
        insertDepths(img, pnts, d, minDepth, maxDepth, false);

        for (auto ip : keyFrames[1].interestPoints)
          kpTerrains[1].checkAllSectors(cam->unmap(ip.p.data()), cam, img);

      } else if (debugOutputType == FILLED_DEPTHS) {
        //        if (interpolationType == PLAIN)
        //          kpTerrains[1].drawDensePlainDepths(img, minDepth, maxDepth);
        // KpTerrains[1].draw(img, CV_BLACK);
      }

      cv::imwrite("../../../../test/data/maps/badtri_removed/frame7005.jpg",
                  img);

      cv::Mat tangImg = kpTerrains[1].drawTangentTri(800, 800);
      //      cv::imwrite("../../../../test/data/maps/badtri/frame505tangentTriang.jpg",
      //                  tangImg);

      cv::Mat img2;
      cv::resize(img, img2, cv::Size(), 0.5, 0.5);

      cv::imshow("tangent", tangImg);
      cv::imshow("interpolated", img);
      cv::waitKey();
    }
  }
}

} // namespace fishdso
