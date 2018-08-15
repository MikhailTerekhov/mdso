#include "frontend/frontend.h"
#include "system/cameramodel.h"
#include "system/dsosystem.h"
#include "util/defs.h"
#include "util/settings.h"
#include "util/util.h"
#include <Eigen/Eigen>
#include <fstream>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <string>

using namespace fishdso;

const std::string winname = "debug";

int main(int argc, char **argv) {
  CameraModel cam(1920, 1208, "../../tests/cam/cam0.txt");

  cam.testReproject();

  //  for (int i = 17; i <= 19; ++i) {
  //    settingCameraMapPolyDegree = i;
  //    {
  //      CameraModel camt(1920, 1208, "../../tests/cam/cam0.txt");
  //      std::cout << i << ' ';
  //      camt.testMapPoly();
  //    }
  //  }

  //  cv::namedWindow(winname, cv::WINDOW_KEEPRATIO);
  //  cv::resizeWindow(winname, 1280, 800);
  //  cv::moveWindow(winname, 200, 200);

  DsoSystem dsoSystem(&cam);
  for (int it = 7000; it <= 9000; ++it) {
    char str[200];
    // sprintf(str, "../../tests/TUM0/%05d.jpg", it);
    sprintf(str, "../../tests/cam/20180306_avm_drive/video_first/%09d.jpg", it);
    cv::Mat frameColored = cv::imread(str);

    if (frameColored.data == NULL) {
      std::cout << "frame could not be read\n";
      return 0;
    }

    //    Mat33 K;
    //    K << 800, 0, 850, 0, 600, 600, 0, 0, 1;
    //    cv::Mat frameUndistort;
    //    cam.undistort<cv::Vec3b>(frameColored, frameUndistort, K);

    // dsoSystem.addFrame(frameUndistort);
    dsoSystem.addFrame(frameColored);
    // dsoSystem.showDebug();
    if (cv::waitKey(10) == 27) // esc
      break;
  }

  return 0;
}
