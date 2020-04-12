#include "output/DebugImageDrawer.h"

DEFINE_double(debug_rel_point_size, 0.004,
              "Relative to w+h point size on debug video.");
DEFINE_int32(debug_image_width, 1200,
             "Width of the image with tracking residuals.");
DEFINE_double(debug_max_stddev, 6.0,
              "Max predicted stddev when displaying debug image with stddevs.");

namespace mdso {

DebugImageDrawer::DebugImageDrawer(const std::vector<int> &drawingOrder)
    : baseFrame(nullptr)
    , lastFrame(nullptr)
    , dso(nullptr)
    , cam(nullptr)
    , drawingOrder(drawingOrder) {}

bool DebugImageDrawer::isDrawable() const {
  return baseFrame && lastFrame && residualsDrawer->isDrawable();
}

void DebugImageDrawer::created(DsoSystem *newDso, CameraBundle *newCam,
                               const Settings &newSettings) {
  dso = newDso;
  cam = newCam;
  settings = newSettings;
  camPyr = cam->camPyr(settings.pyramid.levelNum());
  residualsDrawer = std::unique_ptr<TrackingDebugImageDrawer>(
      new TrackingDebugImageDrawer(camPyr.data(), settings.frameTracker,
                                   settings.pyramid, drawingOrder));
  dso->addFrameTrackerObserver(residualsDrawer.get());
}

void DebugImageDrawer::newFrame(const PreKeyFrame &newFrame) {
  lastFrame = &newFrame;
}

void DebugImageDrawer::newBaseFrame(const KeyFrame &newBaseFrame) {
  baseFrame = &newBaseFrame;
}

std::vector<cv::Mat3b> DebugImageDrawer::drawProjDepths(
    const StdVector<Reprojection> &immatures,
    const StdVector<Reprojection> &optimized) const {
  std::vector<cv::Mat3b> projDepths(cam->bundle.size());
  int w = cam->bundle[0].cam.getWidth(), h = cam->bundle[0].cam.getHeight();
  int s = FLAGS_debug_rel_point_size * (w + h) / 2;

  for (int camInd = 0; camInd < cam->bundle.size(); ++camInd)
    projDepths[camInd] =
        cvtBgrToGray3(baseFrame->preKeyFrame->frames[camInd].frameColored);

  for (const Reprojection &reproj : immatures) {
    putSquare(projDepths[reproj.targetCamInd], toCvPoint(reproj.reprojected), s,
              depthCol(reproj.reprojectedDepth, minDepthCol, maxDepthCol),
              cv::FILLED);
  }
  for (const Reprojection &reproj : optimized)
    putSquare(projDepths[reproj.targetCamInd], toCvPoint(reproj.reprojected), s,
              depthCol(reproj.reprojectedDepth, minDepthCol, maxDepthCol),
              cv::FILLED);

  return projDepths;
}

std::vector<cv::Mat3b>
DebugImageDrawer::drawUseful(const std::vector<const KeyFrame *> &keyFrames,
                             const StdVector<Reprojection> &optimized) const {
  StdVector<Reprojection> reprojectionsOnLast =
      Reprojector<OptimizedPoint>(keyFrames.data(), keyFrames.size(),
                                  dso->bodyToWorld(lastFrame->globalFrameNum),
                                  settings.residualPattern.height)
          .reproject();

  Array2d<std::vector<bool>> isUseful(
      boost::extents[keyFrames.size()][cam->bundle.size()]);
  for (int frameInd = 0; frameInd < keyFrames.size(); ++frameInd)
    for (int camInd = 0; camInd < cam->bundle.size(); ++camInd)
      isUseful[frameInd][camInd].resize(
          keyFrames[frameInd]->frames[camInd].optimizedPoints.size(), false);

  for (const Reprojection &reproj : reprojectionsOnLast)
    isUseful[reproj.hostInd][reproj.hostCamInd][reproj.pointInd] = true;

  std::vector<cv::Mat3b> usefulImg(cam->bundle.size());
  for (int camInd = 0; camInd < cam->bundle.size(); ++camInd)
    usefulImg[camInd] =
        cvtBgrToGray3(baseFrame->preKeyFrame->frames[camInd].frameColored);

  for (const Reprojection &reproj : optimized)
    putDot(usefulImg[reproj.targetCamInd], toCvPoint(reproj.reprojected),
           isUseful[reproj.hostInd][reproj.hostCamInd][reproj.pointInd]
               ? CV_GREEN
               : CV_RED);

  return usefulImg;
}

std::vector<cv::Mat3b>
DebugImageDrawer::drawStddevs(const std::vector<const KeyFrame *> &keyFrames,
                              const StdVector<Reprojection> &immatures,
                              const StdVector<Reprojection> &optimized) const {
  int w = cam->bundle[0].cam.getWidth(), h = cam->bundle[0].cam.getHeight();
  int s = FLAGS_debug_rel_point_size * (w + h) / 2;

  double minStddev = std::sqrt(settings.pointTracer.positionVariance /
                               settings.residualPattern.pattern().size());
  std::vector<cv::Mat3b> stddevsImg(cam->bundle.size());
  for (int camInd = 0; camInd < cam->bundle.size(); ++camInd)
    stddevsImg[camInd] =
        cvtBgrToGray3(baseFrame->preKeyFrame->frames[camInd].frameColored);

  for (const Reprojection &reproj : immatures) {
    double stddev = keyFrames[reproj.hostInd]
                        ->frames[reproj.hostCamInd]
                        .immaturePoints[reproj.pointInd]
                        .stddev;
    putSquare(stddevsImg[reproj.targetCamInd], toCvPoint(reproj.reprojected), s,
              depthCol(stddev, minStddev, FLAGS_debug_max_stddev), cv::FILLED);
  }

  for (const Reprojection &reproj : optimized) {
    double stddev = keyFrames[reproj.hostInd]
                        ->frames[reproj.hostCamInd]
                        .optimizedPoints[reproj.pointInd]
                        .stddev;
    putSquare(stddevsImg[reproj.targetCamInd], toCvPoint(reproj.reprojected), s,
              depthCol(stddev, minStddev, FLAGS_debug_max_stddev), cv::FILLED);
  }

  return stddevsImg;
}

cv::Mat3b DebugImageDrawer::draw() {
  CHECK(isDrawable());

  std::vector<const KeyFrame *> keyFrames = dso->getKeyFrames();
  StdVector<Reprojection> immatures =
      Reprojector<ImmaturePoint>(keyFrames.data(), keyFrames.size(),
                                 baseFrame->thisToWorld(),
                                 settings.residualPattern.height)
          .reproject();
  StdVector<Reprojection> optimized =
      Reprojector<OptimizedPoint>(keyFrames.data(), keyFrames.size(),
                                  baseFrame->thisToWorld(),
                                  settings.residualPattern.height)
          .reproject();
  std::vector<cv::Mat3b> depths = drawProjDepths(immatures, optimized);
  std::vector<cv::Mat3b> isUseful = drawUseful(keyFrames, optimized);
  std::vector<cv::Mat3b> stddevs = drawStddevs(keyFrames, immatures, optimized);
  std::vector<cv::Mat3b> residuals = residualsDrawer->drawFinestLevel();

  cv::Mat3b allDepths;
  cv::vconcat(depths.data(), depths.size(), allDepths);
  cv::Mat3b allUseful;
  cv::vconcat(isUseful.data(), isUseful.size(), allUseful);
  cv::Mat3b allStddevs;
  cv::vconcat(stddevs.data(), stddevs.size(), allStddevs);
  cv::Mat3b allResiduals;
  cv::vconcat(residuals.data(), residuals.size(), allResiduals);

  cv::Mat3b everything;
  cv::hconcat(std::vector{allDepths, allUseful, allStddevs, allResiduals},
              everything);
  return everything;
}

} // namespace mdso
