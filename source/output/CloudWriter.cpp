#include "output/CloudWriter.h"

namespace mdso {

CloudWriter::CloudWriter(CameraBundle *cam, const fs::path &outputDirectory,
                         const fs::path &fileName, bool newOutputStddev)
    : cam(cam)
    , outputDirectory(outputDirectory)
    , cloudHolder(outputDirectory / fileName, newOutputStddev)
    , outputStddev(newOutputStddev) {}

void CloudWriter::keyFramesMarginalized(const KeyFrame *marginalized[],
                                        int size) {
  for (int i = 0; i < size; ++i) {
    const KeyFrame *kf = marginalized[i];
    std::vector<Vec3> points;
    std::vector<cv::Vec3b> colors;
    std::vector<double> stddevs;

    for (int j = 0; j < kf->frames.size(); ++j) {
      const KeyFrameEntry &e = kf->frames[j];
      SE3 thisToWorld = kf->thisToWorld() * cam->bundle[j].thisToBody;

      for (const auto &op : e.optimizedPoints) {
        points.push_back(
            thisToWorld *
            (op.depth() * cam->bundle[j].cam.unmap(op.p).normalized()));
        colors.push_back(kf->preKeyFrame->frames[j].frameColored.at<cv::Vec3b>(
            toCvPoint(op.p)));
        stddevs.push_back(op.stddev);
      }
      for (const auto &ip : e.immaturePoints) {
        if (ip.numTraced > 0) {
          points.push_back(
              kf->thisToWorld() *
              (ip.depth * cam->bundle[j].cam.unmap(ip.p).normalized()));
          colors.push_back(
              kf->preKeyFrame->frames[j].frameColored.at<cv::Vec3b>(
                  toCvPoint(ip.p)));
          stddevs.push_back(ip.stddev);
        }
      }
    }

    Timestamp kfnum = kf->preKeyFrame->frames[0].timestamp;
    fs::path kfCloudFname =
        outputDirectory / fs::path("kf" + std::to_string(kfnum) + ".ply");
    if (outputStddev) {
      printInPly(kfCloudFname, points, colors, stddevs);
      cloudHolder.putPoints(points, colors, stddevs);
    } else {
      std::ofstream kfOut(kfCloudFname);
      printInPly(kfOut, points, colors);
    }
  }

  cloudHolder.updatePointCount();
}

void CloudWriter::destructed(const KeyFrame *lastKeyFrames[], int size) {
  keyFramesMarginalized(lastKeyFrames, size);
}

} // namespace mdso
