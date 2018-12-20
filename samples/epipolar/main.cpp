#include "util/defs.h"
#include "system/ImmaturePoint.h"
#include <opencv2/opencv.hpp>

using namespace fishdso;

DEFINE_int32(x, 100, "x-coordinate of the point on base frame");
DEFINE_int32(y, 100, "y-coordinate of the point on base frame");

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  double scale = 604.0;
  Vec2 center(1.58492, 1.07424);
  int unmapPolyDeg = 5;
  VecX unmapPolyCoeffs(unmapPolyDeg, 1);
  unmapPolyCoeffs << 1.14169, -0.203229, -0.362134, 0.351011, -0.147191;
  int width = 1920, height = 1208;
  CameraModel cam(width, height, scale, center, unmapPolyCoeffs);

  cv::Mat image(height, width, CV_8UC3, CV_BLACK);
  PreKeyFrame baseFrame(&cam, image, 0);
  PreKeyFrame refFrame(&cam, image, 1);

  SE3 motion(SO3(), Vec3(0, 1, 0));
  refFrame.worldToThis = motion;

  ImmaturePoint p(&baseFrame, Vec2(double(FLAGS_x), double(FLAGS_y)));
  
  p.traceOn(refFrame, ImmaturePoint::DRAW_EPIPOLE);

  return 0;
}
