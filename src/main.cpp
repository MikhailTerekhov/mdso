#include "frontend/frontend.h"
#include "system/dsosystem.h"
#include "util/defs.h"
#include "util/settings.h"
#include "util/util.h"
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <string>

using namespace fishdso;

const std::string winname = "debug";

int main(int argc, char **argv) {
  cv::namedWindow(winname, cv::WINDOW_KEEPRATIO);
  cv::resizeWindow(winname, 1280, 800);
  cv::moveWindow(winname, 200, 200);

  DsoSystem dsoSystem;

  for (int it = 1; it <= 1499; ++it) {
    char str[200];
    // sprintf(str, "../../tests/TUM0/%05d.jpg", it);
    sprintf(str, "../../tests/cam/20180306_avm_drive/video_first/%09d.jpg", it);
    cv::Mat frameColored = cv::imread(str);

    if (frameColored.data == NULL) {
      std::cout << "frame could not be read\n";
      return 0;
    }

    dsoSystem.addKf(frameColored);
    dsoSystem.showDebug();
    if (cv::waitKey(10) == 27) // esc
      break;
  }

  return 0;
}
