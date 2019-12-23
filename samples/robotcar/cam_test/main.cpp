#include "system/CameraModel.h"
#include <gflags/gflags.h>
#include <iostream>
#include <tbb/parallel_for.h>

using namespace mdso;

using LUT_t = Eigen::Matrix<Vec2, Eigen::Dynamic, Eigen::Dynamic>;

DEFINE_bool(test_centers, false,
            "Do we need to test different centers to find better performance?");

class Pinhole {
public:
  Pinhole(double fx, double fy, double px, double py) {
    K << fx, 0, px, 0, fy, py, 0, 0, 1;
    Kinv = K.inverse();
  }

  Vec2 map(const Vec3 &p) {
    Vec3 homo = K * p;
    return Vec2(homo[0] / homo[2], homo[1] / homo[2]);
  }

  Vec3 unmap(const Vec2 &p) {
    Vec3 homo(p[0], p[1], 1);
    return Kinv * homo;
  }

  Mat33 K, Kinv;
  double fx, fy, px, py;
};

void testCam(const CameraModel &cam) {
  const int testCount = 500;
  int usedCount = 0;
  double sumErr = 0, maxErr = 0;
  Vec2 worstP;
  std::mt19937 mt;
  Vec2 sz(cam.getWidth(), cam.getHeight());
  Vec2 approxC = sz / 2;
  double maxR = approxC.norm() / 2;
  std::uniform_real_distribution<double> x(0, cam.getWidth()),
      y(0, cam.getHeight());
  for (int i = 0; i < testCount; ++i) {
    Vec2 p(x(mt), y(mt));
    if ((p - approxC).norm() > maxR)
      continue;
    usedCount++;
    Vec3 r = cam.unmap(p);
    Vec2 pBack = cam.map(r);
    double err = (p - pBack).norm();
    sumErr += err;
    if (maxErr < err) {
      maxErr = err;
      worstP = p;
    }
  }

  std::cout << "worst p = " << worstP.transpose() << "\n";
  Vec3 wpu = cam.unmap(worstP);
  double r = wpu.head<2>().norm();
  std::cout << "unmapped = " << wpu.transpose() << " (r = " << r
            << " z = " << wpu[2] << " theta = " << std::atan2(r, wpu[2])
            << "\n";
  Vec2 back = cam.map(wpu);
  std::cout << "mapped back = " << back.transpose() << "\n";

  std::cout << "solely CameraModel test:\n";
  std::cout << "avgErr = " << sumErr / usedCount << "\nmaxErr = " << maxErr
            << std::endl;
}

volatile int total_cnt = 0;

class ErrChecker {
public:
  ErrChecker(double errs[], Vec2 centers[], LUT_t &lut, CameraModel &cam,
             Pinhole &pcam, std::ostream *errs_out)
      : errs(errs)
      , centers(centers)
      , cam_ptr(&cam)
      , pcam_ptr(&pcam)
      , lut_ptr(&lut)
      , errs_out(errs_out) {}

  void operator()(const tbb::blocked_range<int> &r) const {
    CHECK(!errs_out || (r.begin() == 0 && r.size() == 1));

    Pinhole &pcam = *pcam_ptr;
    LUT_t &lut = *lut_ptr;
    for (int i = r.begin(); i != r.end(); ++i) {
      CameraModel cam = *cam_ptr;
      cam.setImageCenter(centers[i]);
      double sumErr = 0, maxErr = 0;
      for (int yi = 0; yi < lut.rows(); ++yi)
        for (int xi = 0; xi < lut.cols(); ++xi) {
          double x = xi, y = yi;
          Vec2 repr = cam.map(pcam.unmap(Vec2(x, y)));
          Vec2 lutRepr = lut(yi, xi);
          Vec2 err = repr - lutRepr;
          double errN = err.norm();
          sumErr += errN;
          maxErr = std::max(errN, maxErr);

          if (errs_out)
            (*errs_out) << lutRepr[0] << ' ' << lutRepr[1] << ' ' << err[0]
                        << ' ' << err[1] << '\n';
        }
      errs[i] = sumErr / lut.size();

      ++total_cnt;
      // std::cout << total_cnt << '\n'; cc

      if (errs_out)
        // clang-format off
          std::cout << "Lut comparison:\n"
                       "avg err = " << errs[i] << "\n"
                       "max err = " << maxErr << "\n";
      // clang-format on
    }
  }

private:
  double *errs;
  Vec2 *centers;
  CameraModel *cam_ptr;
  Pinhole *pcam_ptr;
  LUT_t *lut_ptr;
  std::ostream *errs_out;
};

int main(int argc, char *argv[]) {
  std::string usage = "Usage: " + std::string(argv[0]) + R"abacaba( cam lut intr
Where cam names the file with Atan-fisheye camera parameters;
lut names the file with LUT after py/robotcar/calib/transform_lut.py
intr names the file with camera matrix intrinsics from the same SDK;
)abacaba";

  gflags::ParseCommandLineFlags(&argc, &argv, true);
  gflags::SetUsageMessage(usage);
  google::InitGoogleLogging(argv[0]);

  if (argc != 4) {
    std::cout << usage;
    return 0;
  }

  std::ifstream lutF(argv[2]);
  if (!lutF.is_open()) {
    std::cout << "LUT file could not be read\n" << usage;
    return 0;
  }
  int w = 0, h = 0;
  lutF >> w >> h;

  LUT_t lut(w, h);

  for (int i = 0; i < w * h; ++i) {
    double xTo, yTo;
    int xi, yi;
    lutF >> yi >> xi >> yTo >> xTo;
    lut(yi, xi) = Vec2(xTo, yTo);
  }

  Settings::CameraModel set;
  set.unmapPolyDegree = 10;

  CameraModel cam(w, h, std::string(argv[1]), CameraModel::POLY_MAP, set);

  testCam(cam);

  std::ifstream intrinF(argv[3]);
  if (!intrinF.is_open()) {
    std::cout << "intrinsics file could not be read\n" << usage;
    return 0;
  }
  double fx, fy, px, py;
  intrinF >> fx >> fy >> px >> py;

  Pinhole pcam(fx, fy, px, py);

  constexpr int cnt = 100;
  Vec2 centers[cnt];
  double errs[cnt];

  double max_diff = 1;
  std::mt19937 mt;
  std::uniform_real_distribution<double> dc(0, max_diff);
  centers[0] = cam.getImgCenter();
  for (int i = 1; i < cnt; ++i)
    centers[i] = centers[0] + Vec2(dc(mt), dc(mt));

  total_cnt = 0;

  if (FLAGS_test_centers) {
    tbb::parallel_for(tbb::blocked_range<int>(0, cnt),
                      ErrChecker(errs, centers, lut, cam, pcam, nullptr));

    int best_ind = std::min_element(errs, errs + cnt) - errs;
    double best = errs[best_ind];

    std::cout << "best error is on ind " << best_ind
              << "\nwith mCenter = " << centers[best_ind].transpose()
              << "\nand is " << best << std::endl;

    double sumErr = std::accumulate(errs, errs + cnt, 0.0);
    std::cout << "other errs avg = " << sumErr / cnt << std::endl;
  } else {
    std::ofstream errs_out("errs.txt");
    double err = -1;
    Vec2 center = cam.getImgCenter();
    ErrChecker(&err, &center, lut, cam, pcam,
               &errs_out)(tbb::blocked_range<int>(0, 1));
  }

  return 0;
}
