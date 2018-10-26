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
  Vec3 inBase = cam->unmap(baseIP.p).normalized() * baseIP.depthd();
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
      const CameraModel *cam, InterestPoint *interestPoint, const Vec2 &pos,
      KeyFrame *baseKf, KeyFrame *refKf)
      : cam(cam), baseDirection(cam->unmap(pos).normalized()),
        refFrame(refFrame), interestPoint(interestPoint), baseKf(baseKf),
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
  InterestPoint *interestPoint;
  KeyFrame *baseKf;
  KeyFrame *refKf;
};

void BundleAdjuster::adjust() {
  int pointsTotal = 0, pointsOOB = 0, pointsOutliers = 0;

  KeyFrame *secondKeyFrame = nullptr;
  for (auto kf : keyFrames)
    if (kf != firstKeyFrame) {
      secondKeyFrame = kf;
      break;
    }

  SE3 oldMotion = secondKeyFrame->preKeyFrame->worldToThis;

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
    auto affLight = keyFrame->preKeyFrame->lightWorldToThis.data;
    problem.AddParameterBlock(affLight, 2);
    problem.SetParameterLowerBound(affLight, 0, settingMinAffineLigthtA);
    problem.SetParameterUpperBound(affLight, 0, settingMaxAffineLigthtA);
    problem.SetParameterLowerBound(affLight, 1, settingMinAffineLigthtB);
    problem.SetParameterUpperBound(affLight, 1, settingMaxAffineLigthtB);

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

  problem.SetParameterization(
      secondKeyFrame->preKeyFrame->worldToThis.translation().data(),
      new ceres::AutoDiffLocalParameterization<SphericalPlus, 3, 2>(
          new SphericalPlus(
              secondKeyFrame->preKeyFrame->worldToThis.translation())));

  if (FLAGS_fixed_motion_on_first_ba && keyFrames.size() == 2) {
    problem.SetParameterBlockConstant(
        secondKeyFrame->preKeyFrame->worldToThis.translation().data());
    problem.SetParameterBlockConstant(
        secondKeyFrame->preKeyFrame->worldToThis.so3().data());
    problem.SetParameterBlockConstant(
        secondKeyFrame->preKeyFrame->lightWorldToThis.data);
  }

  std::cout << "points on the first = " << firstKeyFrame->interestPoints.size()
            << std::endl;
  std::cout << "points on the second = "
            << secondKeyFrame->interestPoints.size() << std::endl;

  std::map<InterestPoint *, std::vector<DirectResidual *>> residualsFor;

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
          if (baseFrame == secondKeyFrame)
            oobPos.push_back(ip.p);
          else
            oobKf1.push_back(ip.p);
          continue;
        }

        problem.AddParameterBlock(&ip.logInvDepth, 1);
        ordering->AddElementToGroup(&ip.logInvDepth, 0);

        for (int i = 0; i < settingResidualPatternSize; ++i) {
          const Vec2 &pos = ip.p + settingResidualPattern[i];
          DirectResidual *newResidual = new DirectResidual(
              interpolated[baseFrame].get(), interpolated[refFrame].get(), cam,
              &ip, pos, baseFrame, refFrame);

          double gradNorm = baseFrame->gradNorm(toCvPoint(pos));
          double c = settingGreadientWeighingConstant;
          double weight = c / std::hypot(c, gradNorm);
          ceres::LossFunction *lossFunc = new ceres::ScaledLoss(
              new ceres::HuberLoss(settingBAOutlierIntensityDiff), weight,
              ceres::Ownership::TAKE_OWNERSHIP);

          residualsFor[&ip].push_back(newResidual);
          problem.AddResidualBlock(
              new ceres::AutoDiffCostFunction<DirectResidual, 1, 1, 3, 4, 3, 4,
                                              2, 2>(newResidual),
              lossFunc, &ip.logInvDepth,
              baseFrame->preKeyFrame->worldToThis.translation().data(),
              baseFrame->preKeyFrame->worldToThis.so3().data(),
              refFrame->preKeyFrame->worldToThis.translation().data(),
              refFrame->preKeyFrame->worldToThis.so3().data(),
              baseFrame->preKeyFrame->lightWorldToThis.data,
              refFrame->preKeyFrame->lightWorldToThis.data);
        }
      }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.linear_solver_ordering = ordering;
  options.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  std::ofstream ofsReport(FLAGS_output_directory + "/BA_report.txt");
  ofsReport << summary.FullReport() << std::endl;

  SE3 newMotion = secondKeyFrame->preKeyFrame->worldToThis;

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

  std::cout << "aff light:\n" << secondKeyFrame->preKeyFrame->lightWorldToThis;

  auto p = std::minmax_element(
      secondKeyFrame->interestPoints.begin(),
      secondKeyFrame->interestPoints.end(),
      [](auto ip1, auto ip2) { return ip1.depthd() < ip2.depthd(); });
  std::cout << "minmax d = " << p.first->depthd() << ' ' << p.second->depthd()
            << std::endl;

  std::vector<double> depthsVec;
  depthsVec.reserve(secondKeyFrame->interestPoints.size());
  for (const InterestPoint &ip : secondKeyFrame->interestPoints)
    depthsVec.push_back(ip.depthd());
  // setDepthColBounds(depthsVec);

  cv::Mat kfDepths = secondKeyFrame->drawDepthedFrame(minDepth, maxDepth);

  StdVector<Vec2> outliers;
  StdVector<Vec2> badDepth;
  for (InterestPoint &ip : secondKeyFrame->interestPoints) {
    std::vector<double> values =
        reservedVector<double>(residualsFor[&ip].size());
    for (DirectResidual *res : residualsFor[&ip]) {
      double value;
      double &logInvDepth = ip.logInvDepth;
      PreKeyFrame *base = res->baseKf->preKeyFrame.get();
      PreKeyFrame *ref = res->refKf->preKeyFrame.get();
      res->operator()(
          &logInvDepth, base->worldToThis.translation().data(),
          base->worldToThis.so3().data(), ref->worldToThis.translation().data(),
          ref->worldToThis.so3().data(), base->lightWorldToThis.data,
          ref->lightWorldToThis.data, &value);
      values.push_back(value);
    }

    std::sort(values.begin(), values.end());
    double median = values[values.size() / 2];
    if (median > settingBAOutlierIntensityDiff) {
      ip.state = InterestPoint::OUTLIER;
      outliers.push_back(ip.p);
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

  cv::destroyAllWindows();
}
} // namespace fishdso
