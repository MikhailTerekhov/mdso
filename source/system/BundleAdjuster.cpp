#include "system/BundleAdjuster.h"
#include "system/AffineLightTransform.h"
#include "util/defs.h"
#include "util/geometry.h"
#include "util/util.h"
#include <ceres/ceres.h>
#include <ceres/local_parameterization.h>
#include <ceres/cubic_interpolation.h>

namespace fishdso {

BundleAdjuster::BundleAdjuster(CameraModel *cam)
    : cam(cam), firstKeyFrame(nullptr) {}

bool BundleAdjuster::isOOB(const SE3 &worldToBase, const SE3 &worldToRef,
                           const OptimizedPoint &baseOP) {
  Vec3 inBase = cam->unmap(baseOP.p).normalized() * baseOP.depth();
  Vec2 reproj = cam->map(worldToRef * worldToBase.inverse() * inBase);
  return !cam->isOnImage(reproj, settingResidualPatternHeight);
}

void BundleAdjuster::addKeyFrame(KeyFrame *keyFrame) {
  if (firstKeyFrame == nullptr)
    firstKeyFrame = keyFrame;
  keyFrames.insert(keyFrame);
}

struct DirectResidual {
  DirectResidual(
      ceres::BiCubicInterpolator<ceres::Grid2D<unsigned char, 1>> *baseFrame,
      ceres::BiCubicInterpolator<ceres::Grid2D<unsigned char, 1>> *refFrame,
      const CameraModel *cam, OptimizedPoint *optimizedPoint, const Vec2 &pos,
      KeyFrame *baseKf, KeyFrame *refKf)
      : cam(cam), baseDirection(cam->unmap(pos).normalized()),
        refFrame(refFrame), optimizedPoint(optimizedPoint), baseKf(baseKf),
        refKf(refKf) {
    baseFrame->Evaluate(pos[1], pos[0], &baseIntencity);
  }

  template <typename T>
  bool operator()(const T *const logInvDepthP, const T *const baseTransP,
                  const T *const baseRotP, const T *const refTransP,
                  const T *const refRotP, const T *const baseAffP,
                  const T *const refAffP, T *res) const {
    typedef Eigen::Matrix<T, 2, 1> Vec2t;
    typedef Eigen::Matrix<T, 3, 1> Vec3t;
    typedef Eigen::Matrix<T, 3, 3> Mat33t;
    typedef Eigen::Quaternion<T> Quatt;
    typedef Sophus::SE3<T> SE3t;

    Eigen::Map<const Vec3t> baseTransM(baseTransP);
    Vec3t baseTrans(baseTransM);
    Eigen::Map<const Quatt> baseRotM(baseRotP);
    Quatt baseRot(baseRotM);
    SE3t worldToBase(baseRot, baseTrans);

    Eigen::Map<const Vec3t> refTransM(refTransP);
    Vec3t refTrans(refTransM);
    Eigen::Map<const Quatt> refRotM(refRotP);
    Quatt refRot(refRotM);
    SE3t worldToRef(refRot, refTrans);

    const T *baseAffLightP = baseAffP;
    AffineLightTransform<T> baseAffLight(baseAffLightP[0], baseAffLightP[1]);

    const T *refAffLightP = refAffP;
    AffineLightTransform<T> refAffLight(refAffLightP[0], refAffLightP[1]);

    AffineLightTransform<T>::normalizeMultiplier(refAffLight, baseAffLight);

    T depth = ceres::exp(-(*logInvDepthP));
    Vec3t refPos = (worldToRef * worldToBase.inverse()) *
                   (baseDirection.cast<T>() * depth);
    Vec2t refPosMapped = cam->map(refPos.data()).template cast<T>();
    T trackedIntensity;
    refFrame->Evaluate(refPosMapped[1], refPosMapped[0], &trackedIntensity);
    *res = refAffLight(trackedIntensity) - baseAffLight(T(baseIntencity));

    return true;
  }

  const CameraModel *cam;
  Vec3 baseDirection;
  double baseIntencity;
  ceres::BiCubicInterpolator<ceres::Grid2D<unsigned char, 1>> *refFrame;
  OptimizedPoint *optimizedPoint;
  KeyFrame *baseKf;
  KeyFrame *refKf;
};

void BundleAdjuster::adjust(int maxNumIterations) {
  int pointsTotal = 0, pointsOOB = 0, pointsOutliers = 0;

  KeyFrame *secondKeyFrame = nullptr;
  for (auto kf : keyFrames)
    if (kf != firstKeyFrame) {
      secondKeyFrame = kf;
      break;
    }

  SE3 oldMotion = secondKeyFrame->preKeyFrame->worldToThis;

  std::shared_ptr<ceres::ParameterBlockOrdering> ordering(
      new ceres::ParameterBlockOrdering());

  ceres::Problem problem;

  for (KeyFrame *keyFrame : keyFrames) {
    problem.AddParameterBlock(
        keyFrame->preKeyFrame->worldToThis.translation().data(), 3);
    problem.AddParameterBlock(keyFrame->preKeyFrame->worldToThis.so3().data(),
                              4, new ceres::EigenQuaternionParameterization());
    auto affLight = keyFrame->preKeyFrame->lightWorldToThis.data;
    problem.AddParameterBlock(affLight, 2);
    problem.SetParameterLowerBound(affLight, 0, settingMinAffineLigthtA);
    problem.SetParameterUpperBound(affLight, 0, settingMaxAffineLigthtA);
    problem.SetParameterLowerBound(affLight, 1, settingMinAffineLigthtB);
    problem.SetParameterUpperBound(affLight, 1, settingMaxAffineLigthtB);

    if (!FLAGS_optimize_affine_light)
      problem.SetParameterBlockConstant(affLight);

    ordering->AddElementToGroup(
        keyFrame->preKeyFrame->worldToThis.translation().data(), 1);
    ordering->AddElementToGroup(keyFrame->preKeyFrame->worldToThis.so3().data(),
                                1);
    ordering->AddElementToGroup(affLight, 1);
  }

  problem.SetParameterBlockConstant(
      firstKeyFrame->preKeyFrame->worldToThis.translation().data());
  problem.SetParameterBlockConstant(
      firstKeyFrame->preKeyFrame->worldToThis.so3().data());
  problem.SetParameterBlockConstant(
      firstKeyFrame->preKeyFrame->lightWorldToThis.data);

  SE3 worldToFirst = firstKeyFrame->preKeyFrame->worldToThis;
  SE3 worldToSecond = secondKeyFrame->preKeyFrame->worldToThis;
  SE3 secondToFirst = worldToFirst * worldToSecond.inverse();
  Vec3 firstFramePos = worldToFirst.inverse().translation();

  if (FLAGS_fixed_motion_on_first_ba && keyFrames.size() == 2) {
    problem.SetParameterBlockConstant(
        secondKeyFrame->preKeyFrame->worldToThis.translation().data());
    problem.SetParameterBlockConstant(
        secondKeyFrame->preKeyFrame->worldToThis.so3().data());
    problem.SetParameterBlockConstant(
        secondKeyFrame->preKeyFrame->lightWorldToThis.data);
  }

  LOG(INFO) << "points on the first = " << firstKeyFrame->optimizedPoints.size()
            << std::endl;
  LOG(INFO) << "points on the second = "
            << secondKeyFrame->optimizedPoints.size() << std::endl;

  std::map<OptimizedPoint *, std::vector<DirectResidual *>> residualsFor;

  StdVector<Vec2> oobPos;
  StdVector<Vec2> oobKf1;

  for (KeyFrame *baseFrame : keyFrames)
    for (const auto &op : baseFrame->optimizedPoints) {
      problem.AddParameterBlock(&op->logInvDepth, 1);
      problem.SetParameterLowerBound(&op->logInvDepth, 0,
                                     -std::log(settingMaxDepth));
      problem.SetParameterUpperBound(&op->logInvDepth, 0,
                                     -std::log(settingMinDepth));

      ordering->AddElementToGroup(&op->logInvDepth, 0);

      for (KeyFrame *refFrame : keyFrames) {
        if (refFrame == baseFrame)
          continue;
        pointsTotal++;
        if (isOOB(baseFrame->preKeyFrame->worldToThis,
                  refFrame->preKeyFrame->worldToThis, *op)) {
          pointsOOB++;
          if (baseFrame == secondKeyFrame)
            oobPos.push_back(op->p);
          else
            oobKf1.push_back(op->p);
          continue;
        }

        for (int i = 0; i < settingResidualPatternSize; ++i) {
          const Vec2 &pos = op->p + settingResidualPattern[i];
          DirectResidual *newResidual = new DirectResidual(
              &baseFrame->preKeyFrame->framePyr.interpolator(0),
              &refFrame->preKeyFrame->framePyr.interpolator(0), cam, op.get(),
              pos, baseFrame, refFrame);

          double gradNorm = baseFrame->preKeyFrame->gradNorm(toCvPoint(pos));
          double c = settingGradientWeighingConstant;
          double weight = c / std::hypot(c, gradNorm);
          ceres::LossFunction *lossFunc = new ceres::ScaledLoss(
              new ceres::HuberLoss(settingBAOutlierIntensityDiff), weight,
              ceres::Ownership::TAKE_OWNERSHIP);

          residualsFor[op.get()].push_back(newResidual);
          problem.AddResidualBlock(
              new ceres::AutoDiffCostFunction<DirectResidual, 1, 1, 3, 4, 3, 4,
                                              2, 2>(newResidual),
              lossFunc, &op->logInvDepth,
              baseFrame->preKeyFrame->worldToThis.translation().data(),
              baseFrame->preKeyFrame->worldToThis.so3().data(),
              refFrame->preKeyFrame->worldToThis.translation().data(),
              refFrame->preKeyFrame->worldToThis.so3().data(),
              baseFrame->preKeyFrame->lightWorldToThis.data,
              refFrame->preKeyFrame->lightWorldToThis.data);
        }
      }
    }

  // TODO normal way of fixing the scale
  for (const auto &op : firstKeyFrame->optimizedPoints)
    problem.SetParameterBlockConstant(&op->logInvDepth);

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.linear_solver_ordering = ordering;
  // options.minimizer_progress_to_stdout = true;
  options.max_num_iterations = maxNumIterations;
  options.num_threads = FLAGS_num_threads;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  // std::ofstream ofsReport(FLAGS_output_directory + "/BA_report.txt");
  // ofsReport << summary.FullReport() << std::endl;

  SE3 newMotion = secondKeyFrame->preKeyFrame->worldToThis;

  LOG(INFO) << "old motion: "
            << "\ntrans = " << oldMotion.translation().transpose()
            << "\nrot =   " << oldMotion.unit_quaternion().coeffs().transpose()
            << std::endl;
  LOG(INFO) << "new motion: "
            << "\ntrans = " << newMotion.translation().transpose()
            << "\nrot =   " << newMotion.unit_quaternion().coeffs().transpose()
            << std::endl;

  double transCos = oldMotion.translation().normalized().dot(
      newMotion.translation().normalized());
  if (transCos < -1)
    transCos = -1;
  if (transCos > 1)
    transCos = 1;
  LOG(INFO) << "diff angles: "
            << "\ntrans = " << 180.0 / M_PI * std::acos(transCos) << "\nrot = "
            << 180.0 / M_PI *
                   (newMotion.so3().inverse() * oldMotion.so3()).log().norm()
            << std::endl;

  LOG(INFO) << "aff light:\n" << secondKeyFrame->preKeyFrame->lightWorldToThis;

  if (secondKeyFrame->optimizedPoints.size() > 0) {
    auto p = std::minmax_element(secondKeyFrame->optimizedPoints.begin(),
                                 secondKeyFrame->optimizedPoints.end(),
                                 [](const auto &op1, const auto &op2) {
                                   return op1->depth() < op2->depth();
                                 });
    LOG(INFO) << "minmax d = " << (*p.first)->depth() << ' '
              << (*p.second)->depth() << std::endl;
  }

  std::vector<double> depthsVec;
  depthsVec.reserve(secondKeyFrame->optimizedPoints.size());
  for (const auto &op : secondKeyFrame->optimizedPoints)
    depthsVec.push_back(op->depth());
  // setDepthColBounds(depthsVec);

  cv::Mat kfDepths = secondKeyFrame->drawDepthedFrame(minDepthCol, maxDepthCol);

  StdVector<Vec2> outliers;
  StdVector<Vec2> badDepth;
  for (const auto &op : secondKeyFrame->optimizedPoints) {
    if (op->state == OptimizedPoint::OOB)
      continue;

    std::vector<double> values =
        reservedVector<double>(residualsFor[op.get()].size());
    for (DirectResidual *res : residualsFor[op.get()]) {
      double value;
      double &logInvDepth = op->logInvDepth;
      PreKeyFrame *base = res->baseKf->preKeyFrame.get();
      PreKeyFrame *ref = res->refKf->preKeyFrame.get();
      if (res->operator()(&logInvDepth, base->worldToThis.translation().data(),
                          base->worldToThis.so3().data(),
                          ref->worldToThis.translation().data(),
                          ref->worldToThis.so3().data(),
                          base->lightWorldToThis.data,
                          ref->lightWorldToThis.data, &value))
        values.push_back(value);
    }

    if (values.empty()) {
      op->state = OptimizedPoint::OOB;
      continue;
    }

    std::nth_element(values.begin(), values.begin() + values.size() / 2, values.end());
    double median = values[values.size() / 2];
    if (median > settingBAOutlierIntensityDiff) {
      op->state = OptimizedPoint::OUTLIER;
      outliers.push_back(op->p);
    }
  }
  pointsOutliers = outliers.size();

  LOG(INFO) << "BA results:" << std::endl;
  LOG(INFO) << "total points = " << pointsTotal << std::endl;
  LOG(INFO) << "OOB points = " << pointsOOB << std::endl;
  LOG(INFO) << "outlier points = " << pointsOutliers << std::endl;

  SE3 oldFirstToSecond = secondToFirst.inverse();
  SE3 newFirstToSecond = secondKeyFrame->preKeyFrame->worldToThis *
                         firstKeyFrame->preKeyFrame->worldToThis.inverse();
  LOG(INFO) << "BA first to second kf motion change:\n"
            << "translation diff angle = "
            << 180. / M_PI *
                   angle(oldFirstToSecond.translation(),
                         newFirstToSecond.translation())
            << "\nrotation diff = "
            << 180. / M_PI *
                   (oldFirstToSecond.so3() * newFirstToSecond.so3().inverse())
                       .log()
                       .norm()
            << std::endl;

  LOG(INFO) << summary.FullReport() << std::endl;

  for (const Vec2 &p : outliers)
    cv::circle(kfDepths, toCvPoint(p), 3, CV_BLACK, cv::FILLED);
  for (const Vec2 &p : oobPos)
    putCross(kfDepths, toCvPoint(p), 3, CV_BLACK, 2);

  cv::Mat kf1Depthed =
      firstKeyFrame->drawDepthedFrame(minDepthCol, maxDepthCol);
  // for (Vec2 p : oobKf1)
  // putCross(kf1Depthed, toCvPoint(p), 3, CV_BLACK, 2);

  // cv::imshow("first frame", kf1Depthed);
  if (FLAGS_write_files) {
    cv::imwrite(FLAGS_output_directory + "/firstf.jpg", kf1Depthed);

    // cv::imshow("after ba", kfDepths);
    cv::imwrite(FLAGS_output_directory + "/adjusted.jpg", kfDepths);
  }

  // cv::waitKey();

  // cv::destroyAllWindows();
}
} // namespace fishdso
