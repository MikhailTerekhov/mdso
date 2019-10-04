#ifndef INCLUDE_PREKEYFRAME
#define INCLUDE_PREKEYFRAME

#include "system/AffineLightTransform.h"
#include "system/CameraBundle.h"
#include "system/Preprocessor.h"
#include "util/ImagePyramid.h"
#include "util/settings.h"
#include "util/types.h"
#include <opencv2/core.hpp>
#include <sophus/se3.hpp>

namespace mdso {

struct KeyFrame;
struct TrackingResult;
class PreKeyFrameInternals;

struct PreKeyFrame {

  struct FrameEntry {
    FrameEntry(PreKeyFrame *host, int ind, const cv::Mat &_frameColored,
               const cv::Mat1b &frameProcessed, Timestamp timestamp,
               const Settings::Pyramid &pyrSettings);

    PreKeyFrame *host;
    int ind;
    cv::Mat frameColored;
    cv::Mat1d gradX, gradY, gradNorm;
    ImagePyramid framePyr;

    Timestamp timestamp;
    AffLight lightBaseToThis;
  };

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  PreKeyFrame(KeyFrame *baseFrame, CameraBundle *cam,
              Preprocessor *preprocessor, const cv::Mat coloredFrames[],
              int globalFrameNum, Timestamp timestamps[],
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

  inline const SE3 &baseToThis() const {
    CHECK(wasTracked());
    return mBaseToThis;
  }

  inline bool wasTracked() const { return mWasTracked; }

  void setTracked(const TrackingResult &trackingResult);

  std::vector<FrameEntry> frames;

  KeyFrame *baseFrame;
  CameraBundle *cam;
  SE3 baseToThisPredicted;
  int globalFrameNum;

  Settings::Pyramid pyrSettings;

  std::unique_ptr<PreKeyFrameInternals> internals;

private:
  bool mWasTracked;
  SE3 mBaseToThis;
};

} // namespace mdso

#endif
