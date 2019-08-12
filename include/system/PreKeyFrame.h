#ifndef INCLUDE_PREKEYFRAME
#define INCLUDE_PREKEYFRAME

#include "system/AffineLightTransform.h"
#include "system/CameraBundle.h"
#include "util/ImagePyramid.h"
#include "util/settings.h"
#include "util/types.h"
#include <opencv2/core.hpp>
#include <sophus/se3.hpp>

namespace fishdso {

struct KeyFrame;
class PreKeyFrameInternals;

struct PreKeyFrame {

  struct FrameEntry {
    FrameEntry(const cv::Mat &_frameColored, long long timestamp,
               const Settings::Pyramid &pyrSettings);

    cv::Mat frameColored;
    cv::Mat1d gradX, gradY, gradNorm;
    ImagePyramid framePyr;

    long long timestamp;
    AffLight lightBaseToThis;
  };

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  PreKeyFrame(KeyFrame *baseFrame, CameraBundle *cam,
              const cv::Mat coloredFrames[], int globalFrameNum,
              long long timestamps[],
              const Settings::Pyramid &_pyrSettings = {});
  ~PreKeyFrame();

  inline cv::Mat1b &image(int num) {
    CHECK(num >= 0 && num < frames.size());
    return frames[num].framePyr[0];
  }

  inline const cv::Mat1b &image(int num) const {
    CHECK(num >= 0 && num < frames.size());
    return frames[num].framePyr[0];
  }

  static_vector<FrameEntry, Settings::CameraBundle::max_camerasInBundle> frames;

  KeyFrame *baseFrame;
  CameraBundle *cam;
  SE3 baseToThis;
  int globalFrameNum;

  Settings::Pyramid pyrSettings;

  std::unique_ptr<PreKeyFrameInternals> internals;
};

} // namespace fishdso

#endif
