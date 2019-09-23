#include "output/CloudWriter.h"

namespace mdso {

CloudWriter::CloudWriter(CameraBundle *cam, const fs::path &outputDirectory,
                         const fs::path &fileName)
    : cam(cam)
    , outputDirectory(outputDirectory)
    , cloudHolder(outputDirectory / fileName) {}

void CloudWriter::keyFramesMarginalized(const KeyFrame *marginalized[],
                                        int size) {
  for (int i = 0; i < size; ++i) {
    const KeyFrame *kf = marginalized[i];
    std::vector<Vec3> points;
    std::vector<cv::Vec3b> colors;

    for (int j = 0; j < kf->frames.size(); ++j) {
      const KeyFrameEntry &e = kf->frames[j];
      SE3 thisToWorld = kf->thisToWorld * cam->bundle[j].thisToBody;
      for (const auto &op : e.optimizedPoints) {
        points.push_back(
            thisToWorld *
            (op.depth() * cam->bundle[j].cam.unmap(op.p).normalized()));
        colors.push_back(kf->preKeyFrame->frames[j].frameColored.at<cv::Vec3b>(
            toCvPoint(op.p)));
      }
      for (const auto &ip : e.immaturePoints) {
        if (ip.numTraced > 0) {
          points.push_back(
              kf->thisToWorld *
              (ip.depth * cam->bundle[j].cam.unmap(ip.p).normalized()));
          colors.push_back(
              kf->preKeyFrame->frames[j].frameColored.at<cv::Vec3b>(
                  toCvPoint(ip.p)));
        }
      }
    }

    Timestamp kfnum = kf->preKeyFrame->frames[0].timestamp;
    std::ofstream kfOut(outputDirectory /
                        fs::path("kf" + std::to_string(kfnum) + ".ply"));
    printInPly(kfOut, points, colors);
    kfOut.close();
    cloudHolder.putPoints(points, colors);
  }

  cloudHolder.updatePointCount();
}

void CloudWriter::destructed(const KeyFrame *lastKeyFrames[], int size) {
  keyFramesMarginalized(lastKeyFrames, size);
}

} // namespace mdso
