#include "system/FrameTracker.h"
#include "PreKeyFrameInternals.h"
#include "output/FrameTrackerObserver.h"
#include "util/util.h"
#include <ceres/cubic_interpolation.h>
#include <ceres/problem.h>
#include <cmath>

namespace mdso {

struct PointTrackingResidual {
  PointTrackingResidual(
      const Vec3 &pos, double baseIntensity, const SE3 &baseToBody,
      const SE3 &bodyToRef, const CameraModel *camTracked,
      const ceres::BiCubicInterpolator<ceres::Grid2D<unsigned char, 1>>
          *trackedFrame)
      : pos(pos)
      , baseToBody(baseToBody)
      , bodyToRef(bodyToRef)
      , baseIntensity(baseIntensity)
      , camTracked(camTracked)
      , trackedFrame(trackedFrame) {}

  template <typename T>
  bool operator()(const T *const rotP, const T *const transP,
                  const T *const affLightP, T *res) const {
    typedef Eigen::Matrix<T, 2, 1> Vec2t;
    typedef Eigen::Matrix<T, 3, 1> Vec3t;
    typedef Eigen::Matrix<T, 3, 3> Mat33t;
    typedef Eigen::Quaternion<T> Quatt;
    typedef Sophus::SE3<T> SE3t;

    Eigen::Map<const Vec3t> transM(transP);
    Vec3t trans(transM);
    Eigen::Map<const Quatt> rotM(rotP);
    Quatt rot(rotM);
    SE3t motion(rot, trans);
    SE3t baseToBodyT = baseToBody.cast<T>();
    SE3t bodyToRefT = bodyToRef.cast<T>();

    AffineLightTransform<T> affLight(affLightP[0], affLightP[1]);

    Vec3t newPos = bodyToRefT * motion * baseToBodyT * pos.cast<T>();
    Vec2t newPosProj = camTracked->map(newPos.data());

    T trackedIntensity;
    trackedFrame->Evaluate(newPosProj[1], newPosProj[0], &trackedIntensity);
    res[0] = affLight(trackedIntensity) - baseIntensity;

    return true;
  }

  Vec3 pos;
  SE3 baseToBody, bodyToRef;
  double baseIntensity;
  const CameraModel *camTracked;
  const ceres::BiCubicInterpolator<ceres::Grid2D<unsigned char, 1>>
      *trackedFrame;
};

FrameTracker::TrackingResult::TrackingResult(int camNumber)
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
  intencity = img(cvp);
}

FrameTracker::DepthPyramidSlice::Entry::Entry(const DepthedMultiFrame &frame,
                                              const CameraModel &cam,
                                              int levelNum, int cameraNum) {
  const StdVector<DepthedImagePyramid::Point> &oldDepths =
      frame[cameraNum].depths[levelNum];
  points.reserve(oldDepths.size());
  for (const auto &p : oldDepths)
    points.emplace_back(p, cam, frame[cameraNum][levelNum]);
}

FrameTracker::DepthPyramidSlice::DepthPyramidSlice(
    const DepthedMultiFrame &frame, const CameraBundle &cam, int levelNum)
    : mTotalPoints(0) {
  entries.reserve(frame.size());
  for (int cameraNum = 0; cameraNum < frame.size(); ++cameraNum) {
    entries.emplace_back(frame, levelNum, cameraNum);
    mTotalPoints += entries.back().points.size();
  }
}

FrameTracker::DepthPyramidSlice::Entry &
    FrameTracker::DepthPyramidSlice::operator[](int ind) {
  CHECK(ind >= 0 && ind < entries.size());
  return entries[ind];
}

const FrameTracker::DepthPyramidSlice::Entry &
    FrameTracker::DepthPyramidSlice::operator[](int ind) const {
  CHECK(ind >= 0 && ind < entries.size());
  return entries[ind];
}
int FrameTracker::DepthPyramidSlice::totalPoints() const {
  return mTotalPoints;
}

FrameTracker::FrameTracker(CameraBundle camPyr[],
                           const DepthedMultiFrame &baseFrame,
                           std::vector<FrameTrackerObserver *> &observers,
                           const FrameTrackerSettings &_settings)
    : camPyr(camPyr)
    , observers(observers)
    , settings(_settings) {
  baseFrameSlices.reserve(camPyr[0].bundle.size());
  for (int lvl = 0; lvl < settings.pyramid.levelNum(); ++lvl)
    baseFrameSlices.emplace_back(baseFrame, lvl);

  for (FrameTrackerObserver *obs : observers)
    obs->newBaseFrame(baseFrame);
}

FrameTracker::TrackingResult
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

FrameTracker::TrackingResult
FrameTracker::trackPyrLevel(const PreKeyFrame &frame,
                            const TrackingResult &coarseTrackingResult,
                            int pyrLevel) {

  CameraBundle &cam = camPyr[pyrLevel];
  const DepthPyramidSlice &baseSlice = baseFrameSlices[pyrLevel];
  std::vector<cv::Mat1b> trackedImages(cam.bundle.size());
  std::vector<PreKeyFrameInternals::Interpolator_t *> interpolators;

  for (int i = 0; i < cam.bundle.size(); ++i) {
    trackedImages[i] = frame.frames[i].framePyr[pyrLevel];
    interpolators[i] = &frame.internals->frames[i].interpolator(pyrLevel);
  }

  TrackingResult result = coarseTrackingResult;

  ceres::Problem::Options problemOptions;
  problemOptions.loss_function_ownership =
      ceres::Ownership::DO_NOT_TAKE_OWNERSHIP;
  problemOptions.cost_function_ownership =
      ceres::Ownership::DO_NOT_TAKE_OWNERSHIP;
  problemOptions.local_parameterization_ownership =
      ceres::Ownership::DO_NOT_TAKE_OWNERSHIP;
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

  ceres::HuberLoss intencityHuberLoss(settings.intencity.outlierDiff);
  std::vector<ceres::ScaledLoss> weightedLosses;
  weightedLosses.reserve(maxResiduals);

  std::vector<PointTrackingResidual *> residuals;
  std::vector<ceres::AutoDiffCostFunction<PointTrackingResidual, 1, 4, 3, 2>>
      costFunctions;
  residuals.reserve(maxResiduals);
  costFunctions.reserve(maxResiduals);

  int oldWeightedLossesCapacity = weightedLosses.capacity();
  int oldResidualsCapacity = residuals.capacity();
  int oldCostFunctionsCapacity = costFunctions.capacity();

  for (int baseCamNum = 0; baseCamNum < cam.bundle.size(); ++baseCamNum)
    for (int trackedCamNum = 0; trackedCamNum < cam.bundle.size();
         ++trackedCamNum) {
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
          const double c = settings.gradWeighting.c;
          double weight = c / std::hypot(c, p.gradNorm);
          weightedLosses.emplace_back(&intencityHuberLoss, weight,
                                      ceres::Ownership::DO_NOT_TAKE_OWNERSHIP);
          lossFunc = &weightedLosses.back();
        } else
          lossFunc = &intencityHuberLoss;

        // TODO inds in multicamera
        residuals.push_back(new PointTrackingResidual(
            p.ray, p.intencity, cam.bundle[baseCamNum].thisToBody,
            cam.bundle[trackedCamNum].bodyToThis,
            &cam.bundle[trackedCamNum].cam, interpolators[trackedCamNum]));
        costFunctions.emplace_back(residuals.back());
        problem.AddResidualBlock(&costFunctions.back(), lossFunc,
                                 result.baseToTracked.so3().data(),
                                 result.baseToTracked.translation().data(),
                                 result.lightBaseToTracked[0].data);
      }
    }

  CHECK_EQ(oldWeightedLossesCapacity, weightedLosses.capacity());
  CHECK_EQ(oldResidualsCapacity, residuals.capacity());
  CHECK_EQ(oldCostFunctionsCapacity, costFunctions.capacity());

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.num_threads = settings.threading.numThreads;
  // options.minimizer_progress_to_stdout = true;
  // options.max_num_iterations = 10;
  ceres::Solver::Summary summary;

  ceres::Solve(options, &problem, &summary);
  LOG(INFO) << "time (ms) = " << summary.total_time_in_seconds * 1000
            << std::endl;

  LOG(INFO) << summary.BriefReport() << std::endl;

  StdVector<std::pair<Vec2, double>> pointResiduals;
  pointResiduals.reserve(residuals.size());

  for (auto res : residuals) {
    double eval = -1;
    (*res)(result.baseToTracked.unit_quaternion().coeffs().data(),
           result.baseToTracked.translation().data(),
           result.lightBaseToTracked[0].data, &eval);
    Vec2 onTracked = cam.bundle[0].cam.map(result.baseToTracked * res->pos);
    pointResiduals.push_back(std::pair(onTracked, eval));
  }

  for (FrameTrackerObserver *obs : observers)
    obs->levelTracked(pyrLevel, result, pointResiduals.data(),
                      pointResiduals.size());

  return result;
}

} // namespace mdso
