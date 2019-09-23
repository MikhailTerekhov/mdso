#include "output/DebugImageDrawer.h"

DEFINE_double(debug_rel_point_size, 0.004,
              "Relative to w+h point size on debug video.");
DEFINE_int32(debug_image_width, 1200,
             "Width of the image with tracking residuals.");
DEFINE_double(debug_max_stddev, 6.0,
              "Max predicted stddev when displaying debug image with stddevs.");

namespace fishdso {

DebugImageDrawer::DebugImageDrawer()
    : baseFrame(nullptr)
    , lastFrame(nullptr) {}

void DebugImageDrawer::created(DsoSystem *newDso, CameraBundle *newCam,
                               const Settings &newSettings) {
  dso = newDso;
  cam = newCam;
  settings = newSettings;
  camPyr = cam->camPyr(settings.pyramid.levelNum());
  residualsDrawer =
      std::unique_ptr<TrackingDebugImageDrawer>(new TrackingDebugImageDrawer(
          camPyr.data(), settings.frameTracker, settings.pyramid));
  dso->addFrameTrackerObserver(residualsDrawer.get());
}

void DebugImageDrawer::newFrame(const PreKeyFrame &newFrame) {
  lastFrame = &newFrame;
}

void DebugImageDrawer::newBaseFrame(const KeyFrame &newBaseFrame) {
  baseFrame = &newBaseFrame;
}

cv::Mat3b DebugImageDrawer::drawProjDepths(
    const StdVector<Vec2> &optProj, const std::vector<double> &optDepths,
    const StdVector<Vec2> &immProj, const std::vector<ImmaturePoint *> &immRefs,
    const std::vector<double> &immDepths) {
  int w = cam->bundle[0].cam.getWidth(), h = cam->bundle[0].cam.getHeight();
  int s = FLAGS_debug_rel_point_size * (w + h) / 2;

  if (!baseFrame || !lastFrame)
    return cv::Mat3b::zeros(h, w);

  cv::Mat3b result =
      cvtBgrToGray3(baseFrame->preKeyFrame->frames[0].frameColored);
  for (int i = 0; i < immProj.size(); ++i)
    if (immRefs[i]->numTraced > 0)
      putSquare(result, toCvPoint(immProj[i]), s,
                depthCol(immDepths[i], minDepthCol, maxDepthCol), cv::FILLED);
  for (int i = 0; i < optProj.size(); ++i)
    putSquare(result, toCvPoint(optProj[i]), s,
              depthCol(optDepths[i], minDepthCol, maxDepthCol), cv::FILLED);

  return result;
}

cv::Mat3b
DebugImageDrawer::drawUseful(const StdVector<Vec2> &optBaseProj,
                             const std::vector<OptimizedPoint *> &optBaseRefs) {
  int w = cam->bundle[0].cam.getWidth(), h = cam->bundle[0].cam.getHeight();
  int s = FLAGS_debug_rel_point_size * (w + h) / 2;

  if (!baseFrame || !lastFrame)
    return cv::Mat3b::zeros(h, w);

  cv::Mat3b base =
      cvtBgrToGray3(baseFrame->preKeyFrame->frames[0].frameColored);

  StdVector<Vec2> optLastProj(settings.maxOptimizedPoints());
  std::vector<OptimizedPoint *> optLastRefs(settings.maxOptimizedPoints());

  Vec2 *optLastProjData = optLastProj.data();
  OptimizedPoint **optLastRefsData = optLastRefs.data();
  int optLastSize = 0;

  dso->projectOntoFrame<OptimizedPoint>(
      lastFrame->globalFrameNum, &optLastProjData,
      std::make_optional(&optLastRefsData), std::nullopt, std::nullopt,
      &optLastSize);

  optLastProj.resize(optLastSize);
  optLastRefs.resize(optLastSize);

  std::sort(optLastRefs.begin(), optLastRefs.end());

  cv::Mat3b result =
      cvtBgrToGray3(baseFrame->preKeyFrame->frames[0].frameColored);

  for (int i = 0; i < optBaseProj.size(); ++i) {
    cv::Scalar col = std::binary_search(optLastRefs.begin(), optLastRefs.end(),
                                        optBaseRefs[i])
                         ? CV_GREEN
                         : CV_RED;
    putSquare(result, toCvPoint(optBaseProj[i]), s, col, cv::FILLED);
  }

  return result;
}

cv::Mat3b
DebugImageDrawer::drawStddevs(const StdVector<Vec2> &optProj,
                              const std::vector<OptimizedPoint *> &optRefs,
                              const StdVector<Vec2> &immProj,
                              const std::vector<ImmaturePoint *> &immRefs) {
  int w = cam->bundle[0].cam.getWidth(), h = cam->bundle[0].cam.getHeight();
  int s = FLAGS_debug_rel_point_size * (w + h) / 2;

  if (!baseFrame || !lastFrame)
    return cv::Mat3b::zeros(h, w);

  cv::Mat3b result =
      cvtBgrToGray3(baseFrame->preKeyFrame->frames[0].frameColored);
  double minStddev = std::sqrt(settings.pointTracer.positionVariance /
                               settings.residualPattern.pattern().size());
  for (int i = 0; i < immProj.size(); ++i)
    if (immRefs[i]->numTraced > 0) {
      double dev = immRefs[i]->stddev;
      putSquare(result, toCvPoint(immProj[i]), s,
                depthCol(dev, minStddev, FLAGS_debug_max_stddev), cv::FILLED);
    }
  for (int i = 0; i < optProj.size(); ++i) {
    double dev = optRefs[i]->stddev;
    putSquare(result, toCvPoint(optProj[i]), s,
              depthCol(dev, minStddev, FLAGS_debug_max_stddev), cv::FILLED);
  }

  return result;
}

cv::Mat3b DebugImageDrawer::draw() {
  CHECK(cam->bundle.size() == 1) << "Multicamera case is NIY";

  int w = cam->bundle[0].cam.getWidth(), h = cam->bundle[0].cam.getHeight();

  if (!baseFrame || !lastFrame)
    return cv::Mat3b::zeros(2 * h, 2 * w);

  int toResize =
      settings.maxKeyFrames() * settings.keyFrame.immaturePointsNum();
  StdVector<Vec2> immProj(toResize);
  std::vector<double> immDepths(toResize);
  std::vector<ImmaturePoint *> immRefs(toResize);

  Vec2 *immProjData = immProj.data();
  double *immDepthsData = immDepths.data();
  ImmaturePoint **immRefsData = immRefs.data();
  int immSize = 0;

  dso->projectOntoFrame<ImmaturePoint>(
      baseFrame->preKeyFrame->globalFrameNum, &immProjData,
      std::make_optional(&immRefsData), std::nullopt,
      std::make_optional(&immDepthsData), &immSize);

  immProj.resize(immSize);
  immDepths.resize(immSize);
  immRefs.resize(immSize);

  StdVector<Vec2> optProj(settings.maxOptimizedPoints());
  std::vector<double> optDepths(settings.maxOptimizedPoints());
  std::vector<OptimizedPoint *> optRefs(settings.maxOptimizedPoints());

  Vec2 *optProjData = optProj.data();
  double *optDepthsData = optDepths.data();
  OptimizedPoint **optRefsData = optRefs.data();
  int optSize = 0;

  dso->projectOntoFrame<OptimizedPoint>(
      baseFrame->preKeyFrame->globalFrameNum, &optProjData,
      std::make_optional(&optRefsData), std::nullopt,
      std::make_optional(&optDepthsData), &optSize);

  optProj.resize(optSize);
  optDepths.resize(optSize);
  optRefs.resize(optSize);


  cv::Mat3b depths =
      drawProjDepths(optProj, optDepths, immProj, immRefs, immDepths);
  cv::Mat3b usefulImg = drawUseful(optProj, optRefs);
  cv::Mat3b stddevs = drawStddevs(optProj, optRefs, immProj, immRefs);
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
