#include "data/MultiCamReader.h"
#include "data/MultiFovReader.h"

using namespace mdso;

DEFINE_string(mfov_dir, "/shared/datasets/mfov", "MultiFoV dataset location.");
DEFINE_string(mcam_dir, "/shared/datasets/mcam", "MultiCam dataset location.");
DEFINE_string(frames_dir, "/home/mterekhov/data/around",
              "Multi-frame location.");

constexpr int numCams = 4;

// clang-format off
SE3 camToMyBody[numCams] = {
    SE3(
      (Mat33() <<  0,  0,  1,
                  -1,  0,  0,
                   0, -1,  0).finished(),
        Vec3(0.016, 0, -0.001)
    ),
    SE3(
      (Mat33() << -1,  0,  0,
                   0,  0, -1,
                   0, -1,  0).finished(),
      Vec3(0, -0.01, 0.003)
    ),
    SE3(
      (Mat33() <<  0,  0, -1,
                   1,  0,  0,
                   0, -1,  0).finished(),
      Vec3(-0.03, 0, 0)
    ),
    SE3(
      (Mat33() <<  1,  0,  0,
                   0,  0,  1,
                   0, -1,  0).finished(),
        Vec3(0, 0.01, 0.003)
    )
};

SE3 realBodyToMyBody(
    (Mat33() <<  0,  1,  0,
                -1,  0,  0,
                 0,  0,  1).finished(),
    Vec3::Zero());
// clang-format on

std::string camNames[numCams] = {"front", "right", "rear", "left"};

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  MultiFovReader reader(FLAGS_mfov_dir);
  MultiCamReader mcamReader(FLAGS_mcam_dir);

  CameraModel cameraModel = reader.cam().bundle[0].cam;

  SE3 bodyToCam[numCams];
  for (int i = 0; i < numCams; ++i)
    bodyToCam[i] = camToMyBody[i].inverse() * realBodyToMyBody;
  std::vector<CameraModel> cams(numCams, cameraModel);
  CameraBundle cam(bodyToCam, cams.data(), numCams);

  CameraBundle mcamC = mcamReader.cam();

  std::cout << "in dm\n";
  for (int ci = 0; ci < numCams; ++ci) {
    SE3 err = cam.bundle[ci].thisToBody * bodyToCam[ci];
    std::cout << "ci = " << ci
              << " rot err = " << err.so3().log().norm() * 180. / M_PI
              << " trans err = " << err.translation().norm() << std::endl;
    putInMatrixForm(std::cout, bodyToCam[ci].inverse());
    putInMatrixForm(std::cout, cam.bundle[ci].thisToBody);
  }

  std::ofstream ofs("cam_to_body.txt");
  for (int ci = 0; ci < numCams; ++ci)
    putInMatrixForm(ofs, cam.bundle[ci].thisToBody);
  ofs.close();

  fs::path imdir(FLAGS_frames_dir);
  cv::Mat3b images[numCams];
  for (int i = 0; i < numCams; ++i) {
    std::vector<Vec3> points;
    std::vector<cv::Vec3b> colors;
    fs::path imgPath = imdir / (camNames[i] + ".jpg");
    fs::path binPath = imdir / (camNames[i] + ".bin");
    CHECK(fs::exists(imgPath));
    CHECK(fs::exists(binPath));
    images[i] = cv::imread(imgPath.string());
    std::ifstream depthsIfs(binPath);
    for (int y = 0; y < cam.bundle[i].cam.getHeight(); ++y)
      for (int x = 0; x < cam.bundle[i].cam.getWidth(); ++x) {
        float depth;
        depthsIfs.read(reinterpret_cast<char *>(&depth), sizeof(float));
        if (depth > 1e5)
          continue;
        Vec2 p(x, y);
        points.push_back(cam.bundle[i].thisToBody *
                         (depth * cam.bundle[i].cam.unmap(p).normalized()));
        colors.push_back(images[i](toCvPoint(p)));
      }
    std::ofstream plyOfs(camNames[i] + ".ply");
    printInPly(plyOfs, points, colors);
  }

  return 0;
}