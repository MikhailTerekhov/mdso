#include "MultiFovReader.h"
#include "util/types.h"

MultiFovReader::MultiFovReader(const std::string &newMultiFovDir)
    : datasetDir(newMultiFovDir) {
  if (datasetDir.back() == '/')
    datasetDir = datasetDir.substr(0, datasetDir.size() - 1);

  // Our CameraModel is partially compatible with the provided one (affine
  // transformation used in omni_cam is just scaling in our case, but no problem
  // raises since in this dataset no affine transformation is happening). We
  // also compute the inverse polynomial ourselves instead of using the provided
  // one.
  char camFName[256];
  sprintf(camFName, "%s/info/intrinsics.txt", datasetDir.c_str());
  int width, height;
  double unmapPolyCoeffs[5];
  Vec2 center;
  std::ifstream camIfs(camFName);
  if (!camIfs.is_open())
    throw std::runtime_error("could not open camera intrinsics file \"" +
                             std::string(camFName) + "\"");
  camIfs >> width >> height;
  for (int i = 0; i < 5; ++i)
    camIfs >> unmapPolyCoeffs[i];
  VecX ourCoeffs(4);
  ourCoeffs << unmapPolyCoeffs[0], unmapPolyCoeffs[2], unmapPolyCoeffs[3],
      unmapPolyCoeffs[4];
  ourCoeffs *= -1;
  camIfs >> center[0] >> center[1];
  cam = std::unique_ptr<CameraModel>(
      new CameraModel(width, height, 1.0, center, ourCoeffs));

  char posesFName[256];
  sprintf(posesFName, "%s/info/groundtruth.txt", datasetDir.c_str());
  std::ifstream posesIfs(posesFName);
  if (!posesIfs.is_open())
    throw std::runtime_error("could not open ground truth poses file \"" +
                             std::string(posesFName) + "\"");
  worldToFrameGT.push_back(SE3());
  while (!posesIfs.eof()) {
    int frameNum;
    Vec3 trans;
    Eigen::Quaterniond quat;
    posesIfs >> frameNum >> trans[0] >> trans[1] >> trans[2] >> quat.x() >>
        quat.y() >> quat.z() >> quat.w();
    worldToFrameGT.push_back(SE3(quat, trans).inverse());
  }
}

cv::Mat MultiFovReader::getFrame(int globalFrameNum) const {
  char frameFName[256];
  sprintf(frameFName, "%s/data/img/img%04i_0.png", datasetDir.c_str(),
          globalFrameNum);
  cv::Mat result = cv::imread(frameFName);
  if (result.data == NULL)
    throw std::runtime_error("couldn't read frame from \"" +
                             std::string(frameFName) + "\"");
  return result;
}

cv::Mat1d MultiFovReader::getDepths(int globalFrameNum) const {
  char depthsFName[256];
  sprintf(depthsFName, "%s/data/depth/img%04i_0.depth", datasetDir.c_str(),
          globalFrameNum);
  std::ifstream depthsIfs(depthsFName);
  if (!depthsIfs.is_open())
    throw std::runtime_error("could not open depths file \"" +
                             std::string(depthsFName) + "\"");
  cv::Mat1d depths(cam->getHeight(), cam->getWidth());
  for (int y = 0; y < depths.rows; ++y)
    for (int x = 0; x < depths.cols; ++x)
      depthsIfs >> depths(y, x);

  return depths;
}

SE3 MultiFovReader::getWorldToFrameGT(int globalFrameNum) const {
  return worldToFrameGT[globalFrameNum];
}

const StdVector<SE3> &MultiFovReader::getAllWorldToFrameGT() const {
  return worldToFrameGT;
}

int MultiFovReader::getFrameCount() const { return worldToFrameGT.size(); }
