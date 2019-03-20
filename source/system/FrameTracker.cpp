#include "system/FrameTracker.h"
#include "util/defs.h"
#include "util/util.h"
#include <ceres/cubic_interpolation.h>
#include <ceres/problem.h>
#include <chrono>
#include <cmath>

namespace fishdso {

int settingPyrLevelsUnused = 0;
int dbg1 = 0, dbg2 = 0;

FrameTracker::FrameTracker(const StdVector<CameraModel> &camPyr,
                           std::unique_ptr<DepthedImagePyramid> baseFrame)
    : camPyr(camPyr)
    , baseFrame(std::move(baseFrame))
    , displayWidth(camPyr[1].getWidth())
    , displayHeight(camPyr[1].getHeight())
    , lastRmse(INF) {}

std::pair<SE3, AffineLightTransform<double>>
FrameTracker::trackFrame(const ImagePyramid &frame, const SE3 &coarseMotion,
                         const AffineLightTransform<double> &coarseAffLight) {
  SE3 motion = coarseMotion;
  AffineLightTransform<double> affLight;

  for (int i = settingPyrLevels - 1; i >= settingPyrLevelsUnused; --i) {
    LOG(INFO) << "track level #" << i << std::endl;
    std::tie(motion, affLight) =
        trackPyrLevel(camPyr[i], baseFrame->images[i], baseFrame->depths[i],
                      frame.images[i], motion, affLight, i);
  }

  // cv::waitKey();

  return {motion, affLight};
}

struct PointTrackingResidual {
  PointTrackingResidual(
      Vec3 pos, double baseIntensity, const CameraModel *cam,
      ceres::BiCubicInterpolator<ceres::Grid2D<unsigned char, 1>> *trackedFrame)
      : pos(pos)
      , baseIntensity(baseIntensity)
      , cam(cam)
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
    AffineLightTransform<T> affLight(affLightP[0], affLightP[1]);

    Vec3t newPos = motion * pos.cast<T>();
    Vec2t newPosProj = cam->map(newPos.data());

    T trackedIntensity;
    trackedFrame->Evaluate(newPosProj[1], newPosProj[0], &trackedIntensity);
    res[0] = affLight(trackedIntensity) - baseIntensity;

    return true;
  }

  Vec3 pos;
  double baseIntensity;
  const CameraModel *cam;
  ceres::BiCubicInterpolator<ceres::Grid2D<unsigned char, 1>> *trackedFrame;
};

bool isPointTrackable(const CameraModel &cam, const Vec3 &basePos,
                      const SE3 &coarseBaseToCur) {
  Vec3 coarseCurPos = coarseBaseToCur * basePos;
  Vec2 coarseCurOnImg = cam.map(coarseCurPos);
  return cam.isOnImage(coarseCurOnImg, 0);
}

std::pair<SE3, AffineLightTransform<double>> FrameTracker::trackPyrLevel(
    const CameraModel &cam, const cv::Mat1b &baseImg,
    const cv::Mat1d &baseDepths, const cv::Mat1b &trackedImg,
    const SE3 &coarseMotion, const AffineLightTransform<double> &coarseAffLight,
    int pyrLevel) {
  SE3 motion = coarseMotion;
  AffineLightTransform<double> affLight = coarseAffLight;

  cv::Size displSz = cv::Size(displayHeight, displayWidth);
  cv::Mat1b resMask(baseImg.size(), CV_WHITE_BYTE);

  ceres::Grid2D<unsigned char, 1> imgGrid(trackedImg.data, 0, trackedImg.rows,
                                          0, trackedImg.cols);
  ceres::BiCubicInterpolator<ceres::Grid2D<unsigned char, 1>> trackedFrame(
      imgGrid);

  ceres::Problem problem;

  problem.AddParameterBlock(motion.so3().data(), 4,
                            new ceres::EigenQuaternionParameterization());
  problem.AddParameterBlock(motion.translation().data(), 3);

  problem.AddParameterBlock(affLight.data, 2);
  problem.SetParameterLowerBound(affLight.data, 0, settingMinAffineLigthtA);
  problem.SetParameterUpperBound(affLight.data, 0, settingMaxAffineLigthtA);
  problem.SetParameterLowerBound(affLight.data, 1, settingMinAffineLigthtB);
  problem.SetParameterUpperBound(affLight.data, 1, settingMaxAffineLigthtB);

  if (!FLAGS_optimize_affine_light)
    problem.SetParameterBlockConstant(affLight.data);

  std::vector<double> pixUsed;

  ceres::LossFunction *lossFunc =
      new ceres::HuberLoss(settingTrackingOutlierIntensityDiff);

  double pnt[2] = {0.0, 0.0};

  std::vector<PointTrackingResidual *> residuals;
  StdVector<Vec2> gotOutside;

  int pntTotal = 0, pntOutside = 0, pntOutlier = 0;

  for (int y = 0; y < baseImg.rows; ++y)
    for (int x = 0; x < baseImg.cols; ++x)
      if (baseDepths(y, x) > 0) {
        if (!resMask(y, x))
          continue;
        ++pntTotal;
        pnt[0] = x;
        pnt[1] = y;

        Vec3 pos = cam.unmap(pnt).normalized() * baseDepths(y, x);
        if (!isPointTrackable(cam, pos, coarseMotion)) {
          gotOutside.push_back(Vec2(x, y));
          continue;
        }

        pixUsed.push_back(baseImg(y, x));

        ceres::LossFunction *lossFunc = nullptr;
        if (FLAGS_use_grad_weights_on_tracking) {
          double gradNorm = gradNormAt(baseImg, cv::Point(x, y));
          double c = settingGradientWeighingConstant;
          double weight = c / std::hypot(c, gradNorm);
          lossFunc = new ceres::ScaledLoss(
              new ceres::HuberLoss(settingTrackingOutlierIntensityDiff), weight,
              ceres::Ownership::TAKE_OWNERSHIP);
        } else
          lossFunc = new ceres::HuberLoss(settingTrackingOutlierIntensityDiff);

        residuals.push_back(new PointTrackingResidual(
            pos, static_cast<double>(baseImg(y, x)), &cam, &trackedFrame));
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<PointTrackingResidual, 1, 4, 3, 2>(
                residuals.back()),
            lossFunc, motion.so3().data(), motion.translation().data(),
            affLight.data);
      }

  double meanIntens =
      std::accumulate(pixUsed.begin(), pixUsed.end(), 0.0) / pixUsed.size();
  double sumSqDev = 0;
  for (double p : pixUsed)
    sumSqDev += (p - meanIntens) * (p - meanIntens);
  double meanSqDev = std::sqrt(sumSqDev / pixUsed.size());

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.num_threads = FLAGS_num_threads;
  // options.minimizer_progress_to_stdout = true;
  // options.max_num_iterations = 10;
  ceres::Solver::Summary summary;

  ceres::Solve(options, &problem, &summary);
  LOG(INFO) << "time (mcs) = " << summary.total_time_in_seconds * 1000
            << std::endl;

  LOG(INFO) << summary.BriefReport() << std::endl;

  int w = cam.getWidth(), h = cam.getHeight();
  int s = FLAGS_rel_point_size * (w + h) / 2;
  cv::cvtColor(trackedImg, residualsImg[pyrLevel], cv::COLOR_GRAY2BGR);
  for (auto rsd : residuals) {
    double res = -1;
    double robustified[3] = {INF, 0, 0};
    rsd->operator()(motion.unit_quaternion().coeffs().data(),
                    motion.translation().data(), affLight.data, &res);
    lossFunc->Evaluate(res * res, robustified);
    robustified[0] = std::sqrt(robustified[0]);
    Vec2 onTracked = cam.map(motion * rsd->pos);
    if (cam.isOnImage(onTracked, 0))
      putSquare(residualsImg[pyrLevel], toCvPoint(onTracked), s,
                depthCol(robustified[0], 0, FLAGS_debug_max_residual),
                cv::FILLED);
  }

  // cv::imshow("deprhs", dfr);
  // cv::waitKey();

  return {motion, affLight};
}

} // namespace fishdso
