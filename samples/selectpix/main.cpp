#include "util/PixelSelector.h"
#include "util/defs.h"
#include "util/settings.h"
#include "util/util.h"
#include <dirent.h>
#include <glog/logging.h>

using namespace mdso;

constexpr int extSize = 2;
std::string ext[extSize] = {".jpg", ".png"};

bool isImage(const fs::path &fname) {
  std::string extension = fname.extension();
  return std::find(ext, ext + extSize, extension) != ext + extSize;
}

int main(int argc, const char **argv) {
  std::string usage = "Usage: " + std::string(argv[0]) + R"aba( dir"
Where dir names a directory with images with .jpg extension
)aba";
  google::InitGoogleLogging(argv[0]);

  if (argc != 2) {
    std::cout << "Wrong number of arguments!\n" << usage << std::endl;
    return 0;
  }

  PixelSelector pixelSelector;

  fs::path dirname(argv[1]);
  std::vector<fs::path> fnames;
  for (const fs::path &p : fs::directory_iterator(dirname))
    fnames.push_back(p);

  std::sort(fnames.begin(), fnames.end());

  for (const fs::path &fname : fnames) {
    if (isImage(fname)) {
      cv::Mat3b imCol = cv::imread(fname.c_str());
      cv::Mat im = cvtBgrToGray(imCol);
      if (!im.data)
        continue;
      cv::Mat1d gradX, gradY, gradNorm;
      grad(im, gradX, gradY, gradNorm);
      PixelSelector::PointVector points = pixelSelector.select(
          im, gradNorm, Settings::KeyFrame::default_immaturePointsNum, &imCol);

      const int prefW = 1200;
      int prefH = int(double(im.rows) / im.cols * prefW);
      cv::Mat imColR;
      cv::resize(imCol, imColR, cv::Size(prefW, prefH));
      cv::imshow("frame", imColR);
      cv::waitKey();
    }
  }
  return 0;
}
