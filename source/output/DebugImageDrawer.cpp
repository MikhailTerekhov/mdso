#include "output/DebugImageDrawer.h"

namespace fishdso {

DEFINE_double(debug_rel_point_size, 0.004,
              "Relative to w+h point size on debug video.");
DEFINE_int32(debug_image_width, 1200,
             "Width of the image with tracking residuals.");
DEFINE_double(debug_max_stddev, 6.0,
              "Max predicted stddev when displaying debug image with stddevs.");

DebugImageDrawer::DebugImageDrawer()
    : baseFrame(nullptr)
    , lastFrame(nullptr) {}

void DebugImageDrawer::created(DsoSystem *newDso, CameraModel *newCam,
                               const Settings &newSettings) {
  dso = newDso;
  cam = newCam;
  settings = newSettings;
  residualsDrawer = std::unique_ptr<TrackingDebugImageDrawer>(
      new TrackingDebugImageDrawer(cam->camPyr(settings.pyramid.levelNum),
                                   settings.frameTracker, settings.pyramid));
  dso->addFrameTrackerObserver(residualsDrawer.get());
}

void DebugImageDrawer::newFrame(const PreKeyFrame *newFrame) {
  lastFrame = newFrame;
}

void DebugImageDrawer::newKeyFrame(const KeyFrame *newBaseFrame) {
  baseFrame = newBaseFrame;
}

cv::Mat3b DebugImageDrawer::draw() {
  int w = cam->getWidth(), h = cam->getHeight();
  int s = FLAGS_debug_rel_point_size * (w + h) / 2;

  if (!baseFrame || !lastFrame)
    return cv::Mat3b::zeros(h, w);

  cv::Mat3b base = cvtBgrToGray3(baseFrame->preKeyFrame->frameColored);
  StdVector<Vec2> immPt;
  std::vector<double> immD;
  std::vector<ImmaturePoint *> immRef;
  dso->projectOntoBaseKf<ImmaturePoint>(&immPt, &immD, &immRef, nullptr);
  StdVector<Vec2> optPt;
  std::vector<double> optD;
  std::vector<OptimizedPoint *> optRef;
  dso->projectOntoBaseKf<OptimizedPoint>(&optPt, &optD, &optRef, nullptr);

  cv::Mat3b depths = base.clone();
  for (int i = 0; i < immPt.size(); ++i)
    if (immRef[i]->numTraced > 0)
      putSquare(depths, toCvPoint(immPt[i]), s,
                depthCol(immD[i], minDepthCol, maxDepthCol), cv::FILLED);
  for (int i = 0; i < optPt.size(); ++i)
    putSquare(depths, toCvPoint(optPt[i]), s,
              depthCol(optD[i], minDepthCol, maxDepthCol), cv::FILLED);

  cv::Mat3b usefulImg = base.clone();
  SE3 baseToLast =
      lastFrame->worldToThis * baseFrame->preKeyFrame->worldToThis.inverse();
  for (int i = 0; i < optPt.size(); ++i) {
    Vec3 p = optD[i] * cam->unmap(optPt[i]).normalized();
    Vec2 reproj = cam->map(baseToLast * p);
    cv::Scalar col = cam->isOnImage(reproj, settings.residualPattern.height)
                         ? CV_GREEN
                         : CV_RED;
    putSquare(usefulImg, toCvPoint(optPt[i]), s, col, cv::FILLED);
  }

  cv::Mat3b stddevs = base.clone();
  double minStddev = std::sqrt(settings.pointTracer.positionVariance /
                               settings.residualPattern.pattern().size());
  for (int i = 0; i < immPt.size(); ++i) {
    double dev = immRef[i]->stddev;
    if (immRef[i]->numTraced > 0)
      putSquare(stddevs, toCvPoint(immPt[i]), s,
                depthCol(dev, minStddev, FLAGS_debug_max_stddev), cv::FILLED);
  }
  for (int i = 0; i < optPt.size(); ++i) {
    double dev = optRef[i]->stddev;
    putSquare(stddevs, toCvPoint(optPt[i]), s,
              depthCol(dev, minStddev, FLAGS_debug_max_stddev), cv::FILLED);
  }

  cv::Mat3b residuals = residualsDrawer->drawFinestLevel();

  cv::Mat3b row1, row2, resultBig, result;
  cv::hconcat(depths, usefulImg, row1);
  cv::hconcat(stddevs, residuals, row2);
  cv::vconcat(row1, row2, resultBig);
  int newh = double(resultBig.rows) / resultBig.cols * FLAGS_debug_image_width;
  cv::resize(resultBig, result, cv::Size(FLAGS_debug_image_width, newh));
  return result;
}

} // namespace fishdso
