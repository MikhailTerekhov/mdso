#include "util/PixelSelector.h"
#include "util/defs.h"
#include "util/settings.h"
#include "util/util.h"
#include <dirent.h>
#include <glog/logging.h>

using namespace fishdso;

std::string ext[] = {".jpg", ".png"};

bool isImage(const std::string &fname) {
  for (const auto &e : ext)
    if (fname.size() >= e.size() &&
        fname.substr(fname.size() - e.size(), e.size()) == e)
      return true;

  return false;
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

  std::string dirname(argv[1]);
  if (!dirname.empty() && dirname.back() != '/')
    dirname.push_back('/');
  DIR *dir = opendir(argv[1]);
  dirent *curfile;
  std::vector<std::string> fnames;

  while ((curfile = readdir(dir)) != NULL)
    fnames.push_back(std::string(curfile->d_name));

  std::sort(fnames.begin(), fnames.end());

  for (const auto &nm : fnames) {
    std::string fname = dirname + nm;
    if (isImage(fname)) {
      cv::Mat imCol = cv::imread(fname.c_str());
      cv::Mat im = cvtBgrToGray(imCol);
      if (!im.data)
        continue;
      cv::Mat1d gradX, gradY, gradNorm;
      grad(im, gradX, gradY, gradNorm);
      std::vector<cv::Point> points = pixelSelector.select(
          im, gradNorm, Settings::KeyFrame::default_pointsNum, &imCol);

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
