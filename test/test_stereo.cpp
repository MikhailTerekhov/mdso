#include "system/cameramodel.h"
#include "system/stereogeometryestimator.h"
#include <gtest/gtest.h>
#include <memory>
#include <random>
#include <sophus/se3.hpp>

using namespace fishdso;

class StereoPositioningTest : public ::testing::Test {
protected:
  virtual void SetUp() {
    double scale = 604.0;
    Vec2 center(1.58492, 1.07424);
    int unmapPolyDeg = 5;
    VecX unmapPolyCoeffs(unmapPolyDeg, 1);
    unmapPolyCoeffs << 1.14169, -0.203229, -0.362134, 0.351011, -0.147191;
    int width = 1920, height = 1208;
    cam = std::make_unique<CameraModel>(width, height, scale, center,
                                        unmapPolyCoeffs);
  }

  std::unique_ptr<CameraModel> cam;
};

TEST_F(StereoPositioningTest, RandomPointsCoarse) {
  const int npoints = 200;
  const double outlierPart = 0.5;
  const double maxAngle = 20.0 * (M_PI / 180);
  const double cameraMapNoise = 1.0;
  std::vector<Vec3> points(npoints);

  std::mt19937 mt;
  std::uniform_real_distribution<double> xdistr(-10, 10);
  std::uniform_real_distribution<double> ydistr(-10, 10);
  std::uniform_real_distribution<double> zdistr(20, 30);

  std::uniform_int_distribution<> camx(0, cam->getWidth() - 1);
  std::uniform_int_distribution<> camy(0, cam->getHeight() - 1);

  std::uniform_real_distribution<double> angle(-maxAngle, maxAngle);
  std::normal_distribution<double> cameraMapAdd(0, cameraMapNoise);
  //  SO3 rotations[] = {SO3::rotX(angle(mt)),
  //                     SO3::rotX(angle(mt)) * SO3::rotY(angle(mt))};

  std::vector<SO3> rotations = {SO3(),
                                SO3::rotX(angle(mt)),
                                SO3::rotZ(angle(mt)),
                                SO3::rotX(angle(mt)) * SO3::rotY(angle(mt)),
                                SO3::rotX(angle(mt)) * SO3::rotZ(angle(mt)),
                                SO3::rotX(angle(mt)) * SO3::rotY(angle(mt)) *
                                    SO3::rotZ(angle(mt))};
  std::vector<Vec3> translations = {Vec3(0, 0, 10), Vec3(0, 0, -10),
                                    Vec3(0, 10, 0), Vec3(0, -10, 0),
                                    Vec3(10, 0, 0), Vec3(-10, 0, 0)};
  std::vector<SE3> motions;
  for (const SO3 &rot : rotations)
    for (const Vec3 &trans : translations)
      motions.push_back(SE3(rot, trans));

  int it = 0;

  std::cout << "translation direction and rotation errors in degrees:"
            << std::endl;
  for (SE3 mot : motions) {
    //    std::cout << "R = \n"
    //              << mot.rotationMatrix()
    //              << "\n t = " << mot.translation().transpose() << std::endl;

    //    Vec3 t = mot.translation();
    //    Mat33 tCross;
    //    tCross << 0, -t[2], t[1], t[2], 0, -t[0], -t[1], t[0], 0;
    //    Mat33 E = tCross * mot.rotationMatrix();
    //    std::cout << "[t]x =\n" << tCross << "\nE =\n" << E << std::endl;

    std::vector<std::pair<Vec2, Vec2>> imgCorresps;
    imgCorresps.reserve((int(npoints * (1 + outlierPart))));
    for (int i = 0; i < npoints; ++i) {
      Vec3 p(xdistr(mt), ydistr(mt), zdistr(mt));
      Vec3 mp = mot * p;
      double angleP = std::atan2(std::hypot(p[0], p[1]), p[2]);
      double angleMp = std::atan2(std::hypot(mp[0], mp[1]), mp[2]);
      if (angleP < settingInitKeypointsObserveAngle &&
          angleMp < settingInitKeypointsObserveAngle)
        imgCorresps.push_back({cam->map(p.data()), cam->map(mp.data())});
      //        imgCorresps[i] = {
      //            cam->map(p.data()) + Vec2(cameraMapAdd(mt),
      //            cameraMapAdd(mt)), cam->map(mp.data()) +
      //            Vec2(cameraMapAdd(mt), cameraMapAdd(mt))};
    }
    for (int i = npoints; i < int(npoints * (1 + outlierPart)); ++i)
      imgCorresps.push_back({Vec2(double(camx(mt)), double(camy(mt))),
                             Vec2(double(camx(mt)), double(camy(mt)))});

    std::shuffle(imgCorresps.begin(), imgCorresps.end(), mt);

    StereoGeometryEstimator tester(cam.get(), imgCorresps);
    SE3 result = tester.findCoarseMotion();
    //    std::cout << "result is:\nR =\n"
    //              << result.rotationMatrix()
    //              << "\nt = " << result.translation().transpose() <<
    //              std::endl;
    double transErrAngle = std::acos(
        mot.translation().normalized().dot(result.translation().normalized()));
    double relRotAngle = (mot.so3().inverse() * result.so3()).log().norm();
    relRotAngle = std::min(relRotAngle, 2 * M_PI - relRotAngle);
    std::cout << "errs = " << transErrAngle * (180.0 / M_PI) << ' '
              << relRotAngle * (180.0 / M_PI) << "; "
              << int(100.0 * (it + 1) / motions.size()) << "% motions processed"
              << std::endl;
    EXPECT_TRUE(transErrAngle < 5 * (M_PI / 180.0) &&
                relRotAngle < 5 * (M_PI / 180.0))
        << "too big error!"
        << "\ntotal points = " << imgCorresps.size()
        << "\nrot ind = " << it / int(translations.size())
        << " trans ind = " << it % int(translations.size()) << std::endl;

    ++it;
  }
}

TEST_F(StereoPositioningTest, RandomPointsPrecise) {
  const int npoints = 200;
  const double outlierPart = 0;
  const double maxAngle = 20.0 * (M_PI / 180);
  const double cameraMapNoise = 1.0;
  std::vector<Vec3> points(npoints);

  std::mt19937 mt(12352345);
  std::uniform_real_distribution<double> xdistr(-10, 10);
  std::uniform_real_distribution<double> ydistr(-10, 10);
  std::uniform_real_distribution<double> zdistr(10, 20);

  std::uniform_int_distribution<> camx(0, cam->getWidth() - 1);
  std::uniform_int_distribution<> camy(0, cam->getHeight() - 1);

  std::uniform_real_distribution<double> angle(-maxAngle, maxAngle);
  std::normal_distribution<double> cameraMapAdd(0, cameraMapNoise);
  //  SO3 rotations[] = {SO3::rotX(angle(mt)),
  //                     SO3::rotX(angle(mt)) * SO3::rotY(angle(mt))};

  std::vector<SO3> rotations = {SO3(),
                                SO3::rotX(angle(mt)),
                                SO3::rotZ(angle(mt)),
                                SO3::rotX(angle(mt)) * SO3::rotY(angle(mt)),
                                SO3::rotX(angle(mt)) * SO3::rotZ(angle(mt)),
                                SO3::rotX(angle(mt)) * SO3::rotY(angle(mt)) *
                                    SO3::rotZ(angle(mt))};
  std::vector<Vec3> translations = {Vec3(0, 0, 1), Vec3(0, 0, -1),
                                    Vec3(0, 1, 0), Vec3(0, -1, 0),
                                    Vec3(1, 0, 0), Vec3(-1, 0, 0)};
  std::vector<SE3> motions;
  for (const SO3 &rot : rotations)
    for (const Vec3 &trans : translations)
      motions.push_back(SE3(rot, trans));
  int it = 0;

  std::cout << "translation direction and rotation errors in degrees:"
            << std::endl;
  for (SE3 mot : motions) {
    //    std::cout << "R = \n"
    //              << mot.rotationMatrix()
    //              << "\n t = " << mot.translation().transpose() << std::endl;

    //    Vec3 t = mot.translation();
    //    Mat33 tCross;
    //    tCross << 0, -t[2], t[1], t[2], 0, -t[0], -t[1], t[0], 0;
    //    Mat33 E = tCross * mot.rotationMatrix();
    //    std::cout << "[t]x =\n" << tCross << "\nE =\n" << E << std::endl;

    std::vector<std::pair<Vec2, Vec2>> imgCorresps;
    imgCorresps.reserve(int(npoints * (1 + outlierPart)));

    for (int i = 0; i < int(npoints * outlierPart); ++i)
      imgCorresps.push_back({Vec2(double(camx(mt)), double(camy(mt))),
                             Vec2(double(camx(mt)), double(camy(mt)))});

    // int trueOutliers = imgCorresps.size();
    // std::cout << "true outliers = " << trueOutliers << std::endl;

    for (int i = 0; i < npoints; ++i) {
      Vec3 p(xdistr(mt), ydistr(mt), zdistr(mt));
      Vec3 mp = mot * p;
      double angleP = std::atan2(std::hypot(p[0], p[1]), p[2]);
      double angleMp = std::atan2(std::hypot(mp[0], mp[1]), mp[2]);
      if (angleP < settingInitKeypointsObserveAngle &&
          angleMp < settingInitKeypointsObserveAngle)
        imgCorresps.push_back({cam->map(p.data()), cam->map(mp.data())});
      //        imgCorresps[i] = {
      //            cam->map(p.data()) + Vec2(cameraMapAdd(mt),
      //            cameraMapAdd(mt)), cam->map(mp.data()) +
      //            Vec2(cameraMapAdd(mt), cameraMapAdd(mt))};
    }

    std::shuffle(imgCorresps.begin(), imgCorresps.end(), mt);

    //    std::cout << "true inliers = " << imgCorresps.size() - trueOutliers
    //              << std::endl;

    StereoGeometryEstimator tester(cam.get(), imgCorresps);
    SE3 result = tester.findPreciseMotion();
    //    std::cout << "result is:\nR =\n"
    //              << result.rotationMatrix()
    //              << "\nt = " << result.translation().transpose() <<
    //              std::endl;
    double transErrAngle = std::acos(
        mot.translation().normalized().dot(result.translation().normalized()));
    double relRotAngle = (mot.so3().inverse() * result.so3()).log().norm();
    relRotAngle = std::min(relRotAngle, 2 * M_PI - relRotAngle);
    //    std::cout << "errs = " << transErrAngle * (180.0 / M_PI) << ' '
    //              << relRotAngle * (180.0 / M_PI) << ' '
    //              << int(100.0 * it / motions.size()) << "% motions processed"
    //              << std::endl
    //              << std::endl;
    std::cout << "errs = " << transErrAngle * (180.0 / M_PI) << ' '
              << relRotAngle * (180.0 / M_PI) << "; "
              << int(100.0 * (it + 1) / motions.size()) << "% motions processed"
              << std::endl;
    EXPECT_TRUE(transErrAngle < 0.1 * (M_PI / 180.0) &&
                relRotAngle < 0.1 * (M_PI / 180.0))
        << "too big error!"
        << "\nrot ind = " << it / int(translations.size())
        << " trans ind = " << it % int(translations.size()) << std::endl;

    ++it;
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  //::testing::GTEST_FLAG(filter) = "StereoPositioningTest.RandomPointsPrecise";
  return RUN_ALL_TESTS();
}
