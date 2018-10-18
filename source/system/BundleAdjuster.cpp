#include "system/BundleAdjuster.h"
#include "system/AffineLightTransform.h"
#include "util/defs.h"
#include "util/util.h"
#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>

namespace fishdso {

BundleAdjuster::BundleAdjuster(CameraModel *cam)
    : cam(cam), firstKeyFrame(nullptr) {}

bool BundleAdjuster::isOOB(const SE3 &worldToBase, const SE3 &worldToRef,
                           const InterestPoint &baseIP) {
  Vec3 inBase = cam->unmap(baseIP.p).normalized() / baseIP.invDepth;
  Vec2 reproj = cam->map(worldToRef * worldToBase.inverse() * inBase);
  return !(reproj[0] >= 0 && reproj[0] < cam->getWidth() && reproj[1] >= 0 &&
           reproj[1] < cam->getHeight());
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
      const CameraModel *cam, InterestPoint *interestPoint, KeyFrame *baseKf,
      KeyFrame *refKf)
      : cam(cam), refFrame(refFrame), interestPoint(interestPoint),
        baseKf(baseKf), refKf(refKf) {
    for (int i = 0; i < settingResidualPatternSize; ++i) {
      Vec2 pos = interestPoint->p + settingResidualPattern[i];
      baseDirections[i] = cam->unmap(pos).normalized();
      baseFrame->Evaluate(pos[1], pos[0], &baseIntencities[i]);
    }
  }

  template <typename T>
  bool operator()(const T *const invDepthP, const T *const baseTransP,
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

    const T &invDepth = *invDepthP;
    for (int i = 0; i < settingResidualPatternSize; ++i) {
      Vec3t refPos = (worldToRef * worldToBase.inverse()) *
                     (baseDirections[i].cast<T>() / invDepth);
      Vec2t refPosMapped = cam->map(refPos.data()).template cast<T>();
      T trackedIntensity;
      refFrame->Evaluate(refPosMapped[1], refPosMapped[0], &trackedIntensity);
      res[i] =
          refAffLight(trackedIntensity) - baseAffLight(T(baseIntencities[i]));
    }

    return true;
  }

  Vec3 baseDirections[settingResidualPatternSize];
  double baseIntencities[settingResidualPatternSize];
  const CameraModel *cam;
  ceres::BiCubicInterpolator<ceres::Grid2D<unsigned char, 1>> *refFrame;
  InterestPoint *interestPoint;
  KeyFrame *baseKf;
  KeyFrame *refKf;
};

void BundleAdjuster::adjust() {
  int pointsTotal = 0, pointsOOB = 0, pointsOutliers = 0;

  KeyFrame *secondKf = nullptr;
  for (auto kf : keyFrames)
    if (kf != firstKeyFrame)
      secondKf = kf;
  SE3 oldMotion = secondKf->preKeyFrame->worldToThis;

  std::map<KeyFrame *, std::unique_ptr<ceres::Grid2D<unsigned char, 1>>> grid;

  std::map<KeyFrame *, std::unique_ptr<ceres::BiCubicInterpolator<
                           ceres::Grid2D<unsigned char, 1>>>>
      interpolated;
  for (KeyFrame *keyFrame : keyFrames)
    grid[keyFrame] = std::make_unique<ceres::Grid2D<unsigned char, 1>>(
        keyFrame->preKeyFrame->frame().data, 0,
        keyFrame->preKeyFrame->frame().rows, 0,
        keyFrame->preKeyFrame->frame().cols);

  for (KeyFrame *keyFrame : keyFrames)
    interpolated[keyFrame] = std::make_unique<
        ceres::BiCubicInterpolator<ceres::Grid2D<unsigned char, 1>>>(
        *grid[keyFrame]);

  std::shared_ptr<ceres::ParameterBlockOrdering> ordering(
      new ceres::ParameterBlockOrdering());

  ceres::Problem problem;

  for (KeyFrame *keyFrame : keyFrames) {
    problem.AddParameterBlock(
        keyFrame->preKeyFrame->worldToThis.translation().data(), 3);
    problem.AddParameterBlock(keyFrame->preKeyFrame->worldToThis.so3().data(),
                              4, new ceres::EigenQuaternionParameterization());
    problem.AddParameterBlock(keyFrame->preKeyFrame->lightWorldToThis.data, 2);

    ordering->AddElementToGroup(
        keyFrame->preKeyFrame->worldToThis.translation().data(), 1);
    ordering->AddElementToGroup(keyFrame->preKeyFrame->worldToThis.so3().data(),
                                1);
    ordering->AddElementToGroup(keyFrame->preKeyFrame->lightWorldToThis.data,
                                1);
  }

  problem.SetParameterBlockConstant(
      firstKeyFrame->preKeyFrame->worldToThis.translation().data());
  problem.SetParameterBlockConstant(
      firstKeyFrame->preKeyFrame->worldToThis.so3().data());
  problem.SetParameterBlockConstant(
      firstKeyFrame->preKeyFrame->lightWorldToThis.data);
  problem.SetParameterBlockConstant(
      firstKeyFrame->preKeyFrame->lightWorldToThis.data);

  ceres::LossFunction *lossFunc =
      new ceres::HuberLoss(settingBAOutlierIntensityDiff);

  std::cout << "points on the first = " << firstKeyFrame->interestPoints.size()
            << std::endl;
  std::cout << "points on the second = " << secondKf->interestPoints.size()
            << std::endl;

  std::vector<DirectResidual *> residuals;

  std::vector<Vec2> oobPos;
  std::vector<Vec2> oobKf1;

  for (KeyFrame *baseFrame : keyFrames)
    for (InterestPoint &ip : baseFrame->interestPoints)
      for (KeyFrame *refFrame : keyFrames) {
        if (refFrame == baseFrame)
          continue;
        pointsTotal++;
        if (isOOB(baseFrame->preKeyFrame->worldToThis,
                  refFrame->preKeyFrame->worldToThis, ip)) {
          pointsOOB++;
          if (baseFrame == secondKf)
            oobPos.push_back(ip.p);
          else
            oobKf1.push_back(ip.p);
          continue;
        }

        problem.AddParameterBlock(&ip.invDepth, 1);
        problem.SetParameterLowerBound(&ip.invDepth, 0,
                                       1 / settingMaxPointDepth);
        ordering->AddElementToGroup(&ip.invDepth, 0);

        residuals.push_back(new DirectResidual(interpolated[baseFrame].get(),
                                               interpolated[refFrame].get(),
                                               cam, &ip, baseFrame, refFrame));
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<DirectResidual,
                                            settingResidualPatternSize, 1, 3, 4,
                                            3, 4, 2, 2>(residuals.back()),
            lossFunc, &ip.invDepth,
            baseFrame->preKeyFrame->worldToThis.translation().data(),
            baseFrame->preKeyFrame->worldToThis.so3().data(),
            refFrame->preKeyFrame->worldToThis.translation().data(),
            refFrame->preKeyFrame->worldToThis.so3().data(),
            baseFrame->preKeyFrame->lightWorldToThis.data,
            refFrame->preKeyFrame->lightWorldToThis.data);
      }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.linear_solver_ordering = ordering;
  options.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  std::ofstream ofsReport(FLAGS_output_directory + "/BA_report.txt");
  ofsReport << summary.FullReport() << std::endl;

  SE3 newMotion = secondKf->preKeyFrame->worldToThis;

  std::cout << "old motion: "
            << "\ntrans = " << oldMotion.translation().transpose()
            << "\nrot =   " << oldMotion.unit_quaternion().coeffs().transpose()
            << std::endl;
  std::cout << "new motion: "
            << "\ntrans = " << newMotion.translation().transpose()
            << "\nrot =   " << newMotion.unit_quaternion().coeffs().transpose()
            << std::endl;

  double transCos = oldMotion.translation().normalized().dot(
      newMotion.translation().normalized());
  if (transCos < -1)
    transCos = -1;
  if (transCos > 1)
    transCos = 1;
  std::cout << "diff angles: "
            << "\ntrans = " << 180.0 / M_PI * std::acos(transCos) << "\nrot = "
            << 180.0 / M_PI *
                   (newMotion.so3().inverse() * oldMotion.so3()).log().norm()
            << std::endl;

  std::cout << "aff light:\n" << secondKf->preKeyFrame->lightWorldToThis;

  auto p = std::minmax_element(
      secondKf->interestPoints.begin(), secondKf->interestPoints.end(),
      [](auto ip1, auto ip2) { return 1 / ip1.invDepth < 1 / ip2.invDepth; });
  std::cout << "minmax d = " << 1 / p.first->invDepth << ' '
            << 1 / p.second->invDepth << std::endl;

  std::vector<double> depthsVec;
  depthsVec.reserve(secondKf->interestPoints.size());
  for (const InterestPoint &ip : secondKf->interestPoints)
    depthsVec.push_back(1 / ip.invDepth);
  // setDepthColBounds(depthsVec);

  cv::Mat kfDepths = secondKf->drawDepthedFrame(minDepth, maxDepth);

  StdVector<Vec2> outliers;
  StdVector<Vec2> badDepth;
  outliers.reserve(residuals.size());
  badDepth.reserve(residuals.size());
  for (auto res : residuals) {
    if (res->baseKf != secondKf)
      continue;
    double values[settingResidualPatternSize];
    double &invDepth = res->interestPoint->invDepth;
    PreKeyFrame *base = res->baseKf->preKeyFrame.get();
    PreKeyFrame *ref = res->refKf->preKeyFrame.get();
    if (res->operator()(&invDepth, base->worldToThis.translation().data(),
                        base->worldToThis.so3().data(),
                        ref->worldToThis.translation().data(),
                        ref->worldToThis.so3().data(),
                        base->lightWorldToThis.data, ref->lightWorldToThis.data,
                        values)) {
      std::sort(values, values + settingResidualPatternSize);
      double median = values[settingResidualPatternSize / 2];
      if (median > settingBAOutlierIntensityDiff) {
        res->interestPoint->state = InterestPoint::OUTLIER;
        outliers.push_back(res->interestPoint->p);
      }
    }
  }
  pointsOutliers = outliers.size();

  LOG(INFO) << "BA results:" << std::endl;
  LOG(INFO) << "total points = " << pointsTotal << std::endl;
  LOG(INFO) << "OOB points = " << pointsOOB << std::endl;
  LOG(INFO) << "outlier points = " << pointsOutliers << std::endl;

  for (Vec2 p : outliers)
    cv::circle(kfDepths, toCvPoint(p), 3, CV_BLACK, cv::FILLED);
  for (Vec2 p : oobPos)
    putCross(kfDepths, toCvPoint(p), CV_BLACK, 3, 2);

  cv::Mat kf1Depthed = firstKeyFrame->drawDepthedFrame(minDepth, maxDepth);
  // for (Vec2 p : oobKf1)
  // putCross(kf1Depthed, toCvPoint(p), CV_BLACK, 3, 2);

  cv::imshow("first frame", kf1Depthed);
  cv::imwrite(FLAGS_output_directory + "/firstf.jpg", kf1Depthed);

  cv::imshow("after ba", kfDepths);
  cv::imwrite(FLAGS_output_directory + "/adjusted.jpg", kfDepths);
  cv::waitKey();
}
} // namespace fishdso
