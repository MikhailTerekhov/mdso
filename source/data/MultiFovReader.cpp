#include "data/MultiFovReader.h"
#include "util/types.h"

MultiFovReader::MultiFovReader(const fs::path &newMultiFovDir)
    : datasetDir(newMultiFovDir) {
  // Our CameraModel is partially compatible with the provided one (affine
  // transformation used in omni_cam is just scaling in our case, but no problem
  // raises since in this dataset no affine transformation is happening). We
  // also compute the inverse polynomial ourselves instead of using the provided
  // one.
  fs::path camFName = datasetDir / fs::path("info/intrinsics.txt");
  int width, height;
  double unmapPolyCoeffs[5];
  Vec2 center;
  std::ifstream camIfs(camFName);
  if (!camIfs.is_open())
    throw std::runtime_error("could not open camera intrinsics file \"" +
                             camFName.native() + "\"");
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

  fs::path posesFName = datasetDir / fs::path("info/groundtruth.txt");
  std::ifstream posesIfs(posesFName);
  if (!posesIfs.is_open())
    throw std::runtime_error("could not open ground truth poses file \"" +
                             posesFName.native() + "\"");
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
  char innerName[15];
  snprintf(innerName, 15, "img%04i_0.png", globalFrameNum);
  fs::path frameFName = datasetDir / fs::path("data/img") / fs::path(innerName);
  cv::Mat result = cv::imread(frameFName.native());
  if (result.data == NULL)
    throw std::runtime_error("couldn't read frame from \"" +
                             frameFName.native() + "\"");
  return result;
}

cv::Mat1d MultiFovReader::getDepths(int globalFrameNum) const {
  char innerName[18];
  snprintf(innerName, 18, "img%04i_0.depth", globalFrameNum);
  fs::path depthsFName =
      datasetDir / fs::path("data/depth") / fs::path(innerName);
  std::ifstream depthsIfs(depthsFName);
  if (!depthsIfs.is_open())
    throw std::runtime_error("could not open depths file \"" +
                             depthsFName.native() + "\"");
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
