#include "stereomatcher.h"
#include "../util/defs.h"
#include "../util/settings.h"
#include <RelativePoseEstimator.h>

namespace fishdso {

StereoMatcher::StereoMatcher(CameraModel *cam)
    : cam(cam), hasBaseFrame(false), orb(cv::ORB::create()),
      descriptorMatcher(cv::DescriptorMatcher::create(
          cv::DescriptorMatcher::BRUTEFORCE_HAMMING)) {
  int interestWidth, interestHeight;
  cam->getRectByAngle(settingInitKeypointsObserveAngle, interestWidth,
                      interestHeight);
  rectOfInterest = cv::Rect((cam->getWidth() - interestWidth) / 2,
                            (cam->getHeight() - interestHeight) / 2,
                            interestWidth, interestHeight);
  descriptorsMask = cv::Mat1b(cam->getHeight(), cam->getWidth(), CV_BLACK_BYTE);
  cv::rectangle(descriptorsMask, rectOfInterest, CV_WHITE_BYTE, CV_FILLED);
}

void fishdso::StereoMatcher::addBaseFrame(const cv::Mat &newBaseFrame) {
  baseFrame = newBaseFrame;
  orb->detectAndCompute(baseFrame, descriptorsMask, baseFrameKeyPoints,
                        baseFrameDescriptors);
  hasBaseFrame = true;
}

void StereoMatcher::createEstimations(const cv::Mat &frame) {
  std::vector<cv::KeyPoint> curFrameKeyPoints;
  cv::Mat curFrameDescriptors;
  orb->detectAndCompute(frame, descriptorsMask, curFrameKeyPoints,
                        curFrameDescriptors);
  std::vector<cv::DMatch> matches;
  descriptorMatcher->match(curFrameDescriptors, baseFrameDescriptors, matches);

  std::sort(matches.begin(), matches.end());

  while (matches.back().distance >
         settingFeatureMatchThreshold * matches.front().distance)
    matches.pop_back();

  std::cout << "total matches = " << matches.size() << std::endl;

  cv::drawMatches(frame, curFrameKeyPoints, baseFrame, baseFrameKeyPoints,
                  matches, dbg);
  cv::Mat resizedDbg;
  cv::resize(dbg, resizedDbg, cv::Size(), 0.5, 0.5);
  cv::imshow("debug", resizedDbg);

  std::vector<std::pair<Vec3, Vec3>> rays;
  std::vector<std::pair<Vec2, Vec2>> projectedRays;
  rays.reserve(matches.size());
  projectedRays.reserve(matches.size());
  Vec2 baseFramePtVec, curFramePtVec;
  Vec3 baseFrameRay, curFrameRay;
  for (int i = 0; i < matches.size(); ++i) {
    cv::Point2f baseFramePt = baseFrameKeyPoints[matches[i].trainIdx].pt;
    cv::Point2f curFramePt = curFrameKeyPoints[matches[i].queryIdx].pt;

    baseFramePtVec[0] = baseFramePt.x;
    baseFramePtVec[1] = baseFramePt.y;
    curFramePtVec[0] = curFramePt.x;
    curFramePtVec[1] = curFramePt.y;

    baseFrameRay = cam->unmap(baseFramePtVec.data());
    curFrameRay = cam->unmap(curFramePtVec.data());

    projectedRays.push_back({baseFramePtVec, curFramePtVec});
    rays.push_back({baseFrameRay, curFrameRay});
  }

  std::vector<char> inliersMask;
  SE3 motion = estimateMotion(rays, projectedRays, inliersMask);

  std::cout << "R =\n"
            << motion.rotationMatrix()
            << "\nt = " << motion.translation().transpose()
            << "\nquat = " << motion.unit_quaternion().w() << " + ("
            << motion.unit_quaternion().vec().transpose() << ")\n";

  std::cout << "rotation angle = "
            << 2 * std::acos((motion.unit_quaternion().w()));

  std::vector<cv::DMatch> myGoodMatches;
  myGoodMatches.reserve(matches.size());
  for (int i = 0; i < matches.size(); ++i)
    if (inliersMask[i])
      myGoodMatches.push_back(matches[i]);

  std::cout << "inlier matches = " << myGoodMatches.size() << std::endl;

  cv::Mat myMatchesDrawn;
  cv::drawMatches(frame, curFrameKeyPoints, baseFrame, baseFrameKeyPoints,
                  myGoodMatches, myMatchesDrawn);
  cv::Mat myMatchesResized;
  cv::resize(myMatchesDrawn, myMatchesResized, cv::Size(), 0.5, 0.5);
  cv::imshow("inlier matches", myMatchesResized);

  cv::waitKey();
}

EIGEN_STRONG_INLINE int StereoMatcher::findInliers(
    const Mat33 &E, const std::vector<std::pair<Vec3, Vec3>> &rays,
    const std::vector<std::pair<Vec2, Vec2>> &projectedRays,
    std::vector<char> &inliersMask) {
  int result = 0;
  Mat33 Et = E.transpose();
  Vec3 norm1, norm2;
  for (int i = 0; i < rays.size(); ++i) {
    auto r = rays[i];
    const auto &pr = projectedRays[i];
    norm1 = Et * r.second;
    norm2 = E * r.first;
    if (norm1.squaredNorm() > 1e-4)
      r.first -= (r.first.dot(norm1) / norm1.squaredNorm()) * norm1;
    if (norm2.squaredNorm() > 1e-4)
      r.second -= (r.second.dot(norm2) / norm2.squaredNorm()) * norm2;

    double err1 = (cam->map(r.first.data()) - pr.first).squaredNorm();
    double err2 = (cam->map(r.second.data()) - pr.second).squaredNorm();
    if (std::min(err1, err2) < settingEssentialReprojErrThreshold) {
      ++result;
      inliersMask[i] = true;
    } else {
      inliersMask[i] = false;
    }
  }
  return result;
}

EIGEN_STRONG_INLINE SE3 StereoMatcher::extractMotion(
    const Mat33 &E, const std::vector<std::pair<Vec3, Vec3>> &rays,
    std::vector<char> &inliersMask, int &newInliers) {
  Eigen::JacobiSVD<Mat33> svdE(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Mat33 U = svdE.matrixU();
  Mat33 V = svdE.matrixV();

  U *= U.determinant();
  V *= V.determinant();

  Vec3 t = U.block<3, 1>(0, 2);
  Mat33 W;
  W << 0, -1, 0, 1, 0, 0, 0, 0, 1;

  Mat33 R1 = U * W * V.transpose();
  Mat33 R2 = U * W.transpose() * V.transpose();

  SE3 solutions[4] = {SE3(R1, t), SE3(R1, -t), SE3(R2, t), SE3(R2, -t)};

  Mat33 tCross;
  tCross << 0, -t[2], t[1], t[2], 0, -t[0], -t[1], t[0], 0;

  int bestFrontPointsNum = 0;
  SE3 bestSol;
  std::vector<char> bestInliersMask(inliersMask.size());
  std::vector<char> curInliersMask(inliersMask.size());
  for (SE3 sol : solutions) {
    int curFrontPointsNum = 0;
    for (int i = 0; i < rays.size(); ++i) {
      if (!inliersMask[i])
        continue;
      Mat32 A;
      A.block<3, 1>(0, 0) = sol.so3() * rays[i].first;
      A.block<3, 1>(0, 1) = -rays[i].second;

      Vec2 coefs = A.fullPivHouseholderQr().solve(-sol.translation());
      //      std::cout << "err = " << (A * coefs + sol.translation()).norm()
      //                << std::endl;
      if (coefs[0] > 0 && coefs[1] > 0) {
        ++curFrontPointsNum;
        curInliersMask[i] = true;
      } else {
        curInliersMask[i] = false;
      }
    }
    if (curFrontPointsNum > bestFrontPointsNum) {
      bestFrontPointsNum = curFrontPointsNum;
      bestSol = sol;
      std::swap(curInliersMask, bestInliersMask);
    }
  }
  std::swap(inliersMask, bestInliersMask);
  newInliers = bestFrontPointsNum;
  return bestSol;
}

SE3 StereoMatcher::estimateMotion(
    std::vector<std::pair<Vec3, Vec3>> &rays,
    const std::vector<std::pair<Vec2, Vec2>> &projectedRays,
    std::vector<char> &inliersMask) {

  static relative_pose::GeneralizedCentralRelativePoseEstimator<double> est;
  const int N = settingEssentialMinimalSolveN;
  const double p = settingEssentialSuccessProb;
  double q = settingOrbInlierProb;

  SE3 bestMotion;

  int hypotesisInd[N];
  std::pair<Vec3 *, Vec3 *> hypotesis[N];
  Mat33 results[10];
  int bestInliers = -1;
  int iterNum = int(std::log(1 - p) / std::log(1 - std::pow(q, N)));
  for (int it = 0; it < iterNum; ++it) {
    for (int i = 0; i < N; ++i)
      hypotesisInd[i] = rand() % rays.size();
    std::sort(hypotesisInd, hypotesisInd + N);
    bool isRepeated = false;
    for (int i = 0; i + 1 < N; ++i)
      if (hypotesisInd[i] == hypotesisInd[i + 1])
        isRepeated = true;
    if (isRepeated)
      continue;
    for (int i = 0; i < N; ++i)
      hypotesis[i] = std::make_pair<Vec3 *, Vec3 *>(
          &rays[hypotesisInd[i]].first, &rays[hypotesisInd[i]].second);

    int foundN = est.estimate(hypotesis, N, results);
    int maxInliers = 0, maxInliersInd = 0;

    //    std::cout << "resulting norms = ";
    //    for (int i = 0; i < foundN; ++i)
    //      std::cout << results[i].norm() << ' ';
    //    std::cout << std::endl;

    std::vector<char> curInliersMask(rays.size()), bestInliersMask(rays.size());
    for (int i = 0; i < foundN; ++i) {
      int inliers =
          findInliers(results[i], rays, projectedRays, curInliersMask);
      if (inliers > maxInliers) {
        maxInliers = inliers;
        maxInliersInd = i;
        std::swap(bestInliersMask, curInliersMask);
      }
    }
    SE3 curMotion = extractMotion(results[maxInliersInd], rays, bestInliersMask,
                                  maxInliers);
    if (bestInliers < maxInliers) {
      bestInliers = maxInliers;
      bestMotion = curMotion;
      std::swap(inliersMask, bestInliersMask);
    }

    if (q < double(maxInliers) / rays.size()) {
      q = double(maxInliers) / rays.size();
      iterNum = int(std::log(1 - p) / std::log(1 - std::pow(q, N)));
    }
  }

  return bestMotion;
}

} // namespace fishdso
