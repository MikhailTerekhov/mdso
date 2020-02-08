#include "system/FrameTracker.h"
#include "PreKeyFrameEntryInternals.h"
#include "output/FrameTrackerObserver.h"
#include "util/util.h"
#include <ceres/cubic_interpolation.h>
#include <ceres/problem.h>
#include <cmath>

namespace mdso {

struct PointTrackingResidual {
  PointTrackingResidual(
      const Vec3 &posImage, double baseIntensity, const SE3 &baseToBody,
      const SE3 &bodyToRef, const AffLight &affLightBaseToRef,
      CameraModel *camTracked,
      const PreKeyFrameEntryInternals::Interpolator_t *trackedFrame)
      : posBody(baseToBody * posImage)
      , bodyToRef(bodyToRef)
      , affLightBaseToRef(affLightBaseToRef)
      , baseIntensity(baseIntensity)
      , camTracked(camTracked)
      , trackedFrame(trackedFrame) {}

  template <typename T>
  bool operator()(const T *const rotP, const T *const transP,
                  const T *const affLightP, T *res) const {
    using Vec2t = Eigen::Matrix<T, 2, 1>;
    using Vec3t = Eigen::Matrix<T, 3, 1>;
    using Mat33t = Eigen::Matrix<T, 3, 3>;
    using Quatt = Eigen::Quaternion<T>;
    using SE3t = Sophus::SE3<T>;

    Eigen::Map<const Vec3t> transM(transP);
    Vec3t trans(transM);
    Eigen::Map<const Quatt> rotM(rotP);
    Quatt rot(rotM);
    SE3t motion(rot, trans);
    SE3t bodyToRefT = bodyToRef.cast<T>();

    AffineLightTransform<T> affLightRefToRefNew(affLightP[0], affLightP[1]);
    AffineLightTransform<T> affLight =
        affLightRefToRefNew * affLightBaseToRef.cast<T>();

    Vec3t newPos = bodyToRefT * motion * posBody.cast<T>();
    Vec2t newPosProj = camTracked->map(newPos.data());

    T trackedIntensity;
    trackedFrame->Evaluate(newPosProj[1], newPosProj[0], &trackedIntensity);
    res[0] = affLight(trackedIntensity) - baseIntensity;

    return true;
  }

  Vec3 posBody;
  SE3 bodyToRef;
  AffLight affLightBaseToRef;
  double baseIntensity;
  const CameraModel *camTracked;
  const PreKeyFrameEntryInternals::Interpolator_t *trackedFrame;
};

TrackingResult::TrackingResult(int camNumber)
    : lightBaseToTracked(camNumber) {}

FrameTracker::DepthPyramidSlice::Entry::Point::Point(
    const DepthedImagePyramid::Point &p, const CameraModel &cam,
    const cv::Mat1b &img)
    : p(p.p)
    , depth(p.depth)
    , ray(cam.unmap(Point::p).normalized() * depth) {
  cv::Point cvp = toCvPoint(Point::p);
  CHECK(Eigen::AlignedBox2i(Vec2i(1, 1), Vec2i(img.cols - 2, img.rows - 2))
            .contains(toVec2i(cvp)));
  gradNorm = gradNormAt(img, cvp);
  intensity = img(cvp);
}

FrameTracker::DepthPyramidSlice::Entry::Entry(const DepthedMultiFrame &frame,
                                              const CameraModel &cam,
                                              int levelNum, int cameraNum) {
  const StdVector<DepthedImagePyramid::Point> &oldDepths =
      frame[cameraNum].depths[levelNum];
  points.reserve(oldDepths.size());
  for (const auto &p : oldDepths) {
    const cv::Mat1b &img = frame[cameraNum][levelNum];
    cv::Point cvp = toCvPoint(p.p);
    if (Eigen::AlignedBox2i(Vec2i(1, 1), Vec2i(img.cols - 2, img.rows - 2))
            .contains(toVec2i(cvp)))
      points.emplace_back(p, cam, img);
  }
}

FrameTracker::DepthPyramidSlice::DepthPyramidSlice(
    const DepthedMultiFrame &frame, const CameraBundle &cam, int levelNum)
    : mTotalPoints(0) {
  entries.reserve(frame.size());
  for (int cameraNum = 0; cameraNum < frame.size(); ++cameraNum) {
    entries.emplace_back(frame, cam.bundle[cameraNum].cam, levelNum, cameraNum);
    mTotalPoints += entries.back().points.size();
  }
}

FrameTracker::DepthPyramidSlice::Entry &FrameTracker::DepthPyramidSlice::
operator[](int ind) {
  CHECK(ind >= 0 && ind < entries.size());
  return entries[ind];
}

const FrameTracker::DepthPyramidSlice::Entry &FrameTracker::DepthPyramidSlice::
operator[](int ind) const {
  CHECK(ind >= 0 && ind < entries.size());
  return entries[ind];
}
int FrameTracker::DepthPyramidSlice::totalPoints() const {
  return mTotalPoints;
}

FrameTracker::FrameTracker(CameraBundle camPyr[],
                           const DepthedMultiFrame &baseFrame,
                           const KeyFrame &baseFrameAsKf,
                           std::vector<FrameTrackerObserver *> &observers,
                           const FrameTrackerSettings &_settings)
    : camPyr(camPyr)
    , observers(observers)
    , baseAffLightFromTo(camPyr[0].bundle.size(),
                         std::vector<AffLight>(camPyr[0].bundle.size()))
    , settings(_settings) {
  baseFrameSlices.reserve(camPyr[0].bundle.size());
  int camNum = camPyr[0].bundle.size();
  for (int fromInd = 0; fromInd < camNum; ++fromInd)
    for (int toInd = 0; toInd < camNum; ++toInd)
      baseAffLightFromTo[fromInd][toInd] =
          baseFrameAsKf.frames[toInd].lightWorldToThis *
          baseFrameAsKf.frames[fromInd].lightWorldToThis.inverse();

  for (int lvl = 0; lvl < settings.pyramid.levelNum(); ++lvl)
    baseFrameSlices.emplace_back(baseFrame, camPyr[lvl], lvl);

  for (FrameTrackerObserver *obs : observers)
    obs->newBaseFrame(baseFrame);
}

TrackingResult
FrameTracker::trackFrame(const PreKeyFrame &frame,
                         const TrackingResult &coarseTrackingResult) {
  for (FrameTrackerObserver *obs : observers)
    obs->startTracking(frame);

  TrackingResult result = coarseTrackingResult;

  for (int i = settings.pyramid.levelNum() - 1; i >= 0; --i) {
    LOG(INFO) << "track level #" << i << std::endl;
    result = trackPyrLevel(frame, result, i);
  }

  return result;
}

bool isPointTrackable(const CameraModel &camTracked, const Vec3 &basePos,
                      const SE3 &coarseBaseToCur) {
  Vec3 coarseCurPos = coarseBaseToCur * basePos;
  Vec2 coarseCurOnImg = camTracked.map(coarseCurPos);
  return camTracked.isOnImage(coarseCurOnImg, 0);
}

TrackingResult
FrameTracker::trackPyrLevel(const PreKeyFrame &frame,
                            const TrackingResult &coarseTrackingResult,
                            int pyrLevel) {

  CameraBundle &cam = camPyr[pyrLevel];
  const DepthPyramidSlice &baseSlice = baseFrameSlices[pyrLevel];
  std::vector<cv::Mat1b> trackedImages(cam.bundle.size());
  std::vector<PreKeyFrameEntryInternals::Interpolator_t *> interpolators(
      cam.bundle.size());

  for (int i = 0; i < cam.bundle.size(); ++i) {
    trackedImages[i] = frame.frames[i].framePyr[pyrLevel];
    interpolators[i] = &frame.frames[i].internals->interpolator(pyrLevel);
  }

  TrackingResult result = coarseTrackingResult;

  ceres::Problem::Options problemOptions;
  problemOptions.local_parameterization_ownership =
      ceres::DO_NOT_TAKE_OWNERSHIP;
  ceres::Problem problem(problemOptions);

  ceres::EigenQuaternionParameterization quaternionParameterization;

  problem.AddParameterBlock(result.baseToTracked.so3().data(), 4,
                            &quaternionParameterization);
  problem.AddParameterBlock(result.baseToTracked.translation().data(), 3);

  for (auto &affLight : result.lightBaseToTracked) {
    problem.AddParameterBlock(affLight.data, 2);
    problem.SetParameterLowerBound(affLight.data, 0,
                                   settings.affineLight.minAffineLightA);
    problem.SetParameterUpperBound(affLight.data, 0,
                                   settings.affineLight.maxAffineLightA);
    problem.SetParameterLowerBound(affLight.data, 1,
                                   settings.affineLight.minAffineLightB);
    problem.SetParameterUpperBound(affLight.data, 1,
                                   settings.affineLight.maxAffineLightB);
    if (!settings.affineLight.optimizeAffineLight)
      problem.SetParameterBlockConstant(affLight.data);
  }

  int pntTotal = 0;

  const int maxResiduals =
      cam.bundle.size() * cam.bundle.size() * baseSlice.totalPoints();

  std::vector<std::vector<PointTrackingResidual *>> residuals(
      cam.bundle.size());

  for (int baseCamNum = 0; baseCamNum < cam.bundle.size(); ++baseCamNum)
    for (int trackedCamNum = 0; trackedCamNum < cam.bundle.size();
         ++trackedCamNum) {
      if (!settings.frameTracker.doIntercameraReprojection)
        if (baseCamNum != trackedCamNum)
          continue;
      SE3 baseToBody = cam.bundle[baseCamNum].thisToBody;
      SE3 bodyToTracked = cam.bundle[trackedCamNum].bodyToThis;
      SE3 coarseBaseToTracked =
          bodyToTracked * result.baseToTracked * baseToBody;
      for (const auto &p : baseSlice[baseCamNum].points) {

        ++pntTotal;

        if (!isPointTrackable(cam.bundle[trackedCamNum].cam, p.ray,
                              coarseBaseToTracked))
          continue;

        ceres::LossFunction *lossFunc;
        if (settings.frameTracker.useGradWeighting) {
          const double c = settings.residualWeighting.c;
          double weight = c / std::hypot(c, p.gradNorm);
          lossFunc = new ceres::ScaledLoss(
              new ceres::HuberLoss(settings.intensity.outlierDiff), weight,
              ceres::Ownership::TAKE_OWNERSHIP);
        } else
          lossFunc = new ceres::HuberLoss(settings.intensity.outlierDiff);

        residuals[trackedCamNum].push_back(new PointTrackingResidual(
            p.ray, p.intensity, cam.bundle[baseCamNum].thisToBody,
            cam.bundle[trackedCamNum].bodyToThis,
            baseAffLightFromTo[baseCamNum][trackedCamNum],
            &cam.bundle[trackedCamNum].cam, interpolators[trackedCamNum]));
        ceres::CostFunction *newCostFunc =
            new ceres::AutoDiffCostFunction<PointTrackingResidual, 1, 4, 3, 2>(
                residuals[trackedCamNum].back());
        problem.AddResidualBlock(newCostFunc, lossFunc,
                                 result.baseToTracked.so3().data(),
                                 result.baseToTracked.translation().data(),
                                 result.lightBaseToTracked[trackedCamNum].data);
      }
    }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.num_threads = settings.threading.numThreads;
  // options.minimizer_progress_to_stdout = true;
  // options.max_num_iterations = 10;
  ceres::Solver::Summary summary;

  ceres::Solve(options, &problem, &summary);
  LOG(INFO) << "time (ms) = " << summary.total_time_in_seconds * 1000
            << std::endl;

  LOG(INFO) << summary.BriefReport();

  VLOG(1) << summary.FullReport();

  std::vector<StdVector<std::pair<Vec2, double>>> pointResiduals(
      cam.bundle.size());

  for (int trackedCamInd = 0; trackedCamInd < cam.bundle.size();
       ++trackedCamInd) {
    for (auto res : residuals[trackedCamInd]) {
      double eval = -1;
      (*res)(result.baseToTracked.unit_quaternion().coeffs().data(),
             result.baseToTracked.translation().data(),
             result.lightBaseToTracked[trackedCamInd].data, &eval);
      Vec2 onTracked = cam.bundle[trackedCamInd].cam.map(
          res->bodyToRef * result.baseToTracked * res->posBody);
      pointResiduals[trackedCamInd].push_back(std::pair(onTracked, eval));
    }
  }

  for (FrameTrackerObserver *obs : observers)
    obs->levelTracked(pyrLevel, result, pointResiduals);

  return result;
}

} // namespace mdso
