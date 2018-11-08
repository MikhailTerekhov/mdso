#include "KittiReader.h"
#include "util/settings.h"

KittiReader::KittiReader(const std::string &newKittiDir, int sequenceNum,
                         int startFrame)
    : kittiDir(newKittiDir), sequenceNum(sequenceNum) {
  if (kittiDir.back() == '/')
    kittiDir = kittiDir.substr(0, kittiDir.size() - 1);

  char posesFName[256];
  sprintf(posesFName, "%s/dataset/poses/%02i.txt", kittiDir.c_str(),
          sequenceNum);
  std::ifstream posesIfs(posesFName);
  if (!posesIfs.is_open())
    throw std::runtime_error("could not open ground truth poses file \"" +
                             std::string(posesFName) + "\"");
  bool isFirst = true;
  while (!posesIfs.eof()) {
    Mat34 posMat;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 4; ++j)
        posesIfs >> posMat(i, j);
    Mat33 rotMat = posMat.block<3, 3>(0, 0);
    double det = rotMat.determinant();
    if (std::abs(det - 1.0) > 1e-5) 
      std::cout << "bad input rotation determinant = " << det << std::endl;

    Eigen::JacobiSVD<Mat33> svd(rotMat, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Mat33 U = svd.matrixU(), V = svd.matrixV();
    U *= U.determinant();
    V *= V.determinant();
    rotMat = U * V.transpose();

    SE3 worldToThis =
        SE3(rotMat, posMat.block<3, 1>(0, 3)).inverse();
    worldToFrameGT.push_back(worldToThis);
  }
  
  SE3 startToWorld = worldToFrameGT[startFrame].inverse();
  for (SE3 &worldToThis : worldToFrameGT)
    worldToThis = worldToThis * startToWorld;
  double scale = 1.0 / worldToFrameGT[startFrame + settingFirstFramesSkip + 1]
                           .translation()
                           .norm();
  for (SE3 &worldToThis : worldToFrameGT)
    worldToThis.translation() *= scale;

  Mat34 calibMat;
  char calibFName[256];
  sprintf(calibFName, "%s/dataset/sequences/%02i/calib.txt", kittiDir.c_str(),
          sequenceNum);
  std::ifstream calibIfs(calibFName);
  if (!calibIfs.is_open())
    throw std::runtime_error("couldn't open calibration file \"" +
                             std::string(calibFName) + "\"");

  char temp[5];
  calibIfs.read(temp, 4); // read "P0: "
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      calibIfs >> calibMat(i, j);
  std::cout << "calib:\n" << calibMat << std::endl;

  cv::Mat img = getFrame(0);
  std::cout << "w, h = " << img.cols << ' ' << img.rows << std::endl;
  cam = std::unique_ptr<CameraModel>(new CameraModel(
      img.cols, img.rows, calibMat(0, 0), calibMat(0, 2), calibMat(1, 2)));
}

cv::Mat KittiReader::getFrame(int globalFrameNum) {
  char frameFileName[256];
  sprintf(frameFileName, "%s/dataset/sequences/%02i/image_0/%06i.png",
          kittiDir.c_str(), sequenceNum, globalFrameNum);
  cv::Mat result = cv::imread(frameFileName);
  if (result.data == NULL)
    throw std::runtime_error("couldn't read frame from \"" +
                             std::string(frameFileName) + "\"");
  return result;
}

SE3 KittiReader::getWorldToFrameGT(int globalFrameNum) {
  return worldToFrameGT[globalFrameNum];
}
