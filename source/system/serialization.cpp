#include "system/serialization.h"
#include "system/FrameTracker.h"
#include "system/KeyFrame.h"
#include <boost/math/special_functions/nonfinite_num_facets.hpp>

namespace mdso {

std::locale correctLocale() {
  std::locale defaultLocale;
  std::locale tmpLocale(defaultLocale,
                        new boost::math::nonfinite_num_put<char>());
  return std::locale(tmpLocale, new boost::math::nonfinite_num_get<char>());
}

DataSerializer<STORE>::DataSerializer(const fs::path &fname)
    : stream(fname) {
  stream.precision(std::numeric_limits<double>::max_digits10 + 1);
  stream.imbue(correctLocale());
}

DataSerializer<LOAD>::DataSerializer(const fs::path &fname)
    : stream(fname) {
  stream.imbue(correctLocale());
}

template <SerializerMode mode>
PointSerializer<mode>::PointSerializer(const fs::path &pointsFname,
                                       int patternSize)
    : dataSerializer(pointsFname)
    , PS(patternSize) {}

template <SerializerMode mode>
void PointSerializer<mode>::process(RefT<mode, ImmaturePoint> p) {
  dataSerializer.process(p.p);
  for (int i = 0; i < PS; ++i)
    dataSerializer.process(p.baseDirections[i]);
  dataSerializer.process(p.dir);
  for (int i = 0; i < PS; ++i)
    dataSerializer.process(p.baseIntencities[i]);
  for (int i = 0; i < PS; ++i)
    dataSerializer.process(p.baseGrad[i]);
  for (int i = 0; i < PS; ++i)
    dataSerializer.process(p.baseGradNorm[i]);
  dataSerializer.process(p.minDepth);
  dataSerializer.process(p.maxDepth);
  dataSerializer.process(p.depth);
  dataSerializer.process(p.bestQuality);
  dataSerializer.process(p.lastEnergy);
  dataSerializer.process(p.stddev);
  dataSerializer.process(p.state);
}

template <SerializerMode mode>
void PointSerializer<mode>::process(RefT<mode, OptimizedPoint> p) {
  dataSerializer.process(p.p);
  dataSerializer.process(p.dir);
  dataSerializer.process(p.logDepth);
  dataSerializer.process(p.stddev);
  dataSerializer.process(p.minDepth);
  dataSerializer.process(p.maxDepth);
}

PreKeyFrameLoader::PreKeyFrameLoader(const DatasetReader *datasetReader,
                                     const Preprocessor *preprocessor,
                                     CameraBundle *cam, KeyFrame *baseFrame,
                                     const fs::path &preKeyFrameFname,
                                     const Settings::Pyramid &pyramidSettings)
    : datasetReader(datasetReader)
    , preprocessor(preprocessor)
    , cam(cam)
    , baseFrame(baseFrame)
    , preKeyFrameFname(preKeyFrameFname)
    , pyramidSettings(pyramidSettings) {}

std::unique_ptr<PreKeyFrame> PreKeyFrameLoader::load() const {
  DataSerializer<LOAD> dataSerializer(preKeyFrameFname);

  std::vector<Timestamp> timestamps(cam->bundle.size());
  TrackingResult trackingResult(cam->bundle.size());

  dataSerializer.process(trackingResult.baseToTracked);
  for (int i = 0; i < cam->bundle.size(); ++i) {
    dataSerializer.process(timestamps[i]);
    dataSerializer.process(trackingResult.lightBaseToTracked[i]);
  }

  int frameInd = datasetReader->firstTimestampToInd(timestamps[0]);
  auto frame = datasetReader->frame(frameInd);

  std::vector<cv::Mat3b> coloredFrames(cam->bundle.size());
  for (int i = 0; i < cam->bundle.size(); ++i)
    coloredFrames[i] = frame[i].frame;
  std::unique_ptr<PreKeyFrame> preKeyFrame(
      new PreKeyFrame(baseFrame, cam, preprocessor, coloredFrames.data(),
                      frameInd, timestamps.data(), pyramidSettings));
  preKeyFrame->setTracked(trackingResult);
  return preKeyFrame;
}

void PreKeyFrameSaver::store(const fs::path &preKeyFrameFname,
                             const PreKeyFrame &preKeyFrame) {
  CHECK_EQ(preKeyFrame.cam->bundle.size(), 1) << "Multicamera is NIY";

  DataSerializer<STORE> dataSerializer(preKeyFrameFname);
  dataSerializer.process(preKeyFrame.wasTracked() ? preKeyFrame.baseToThis()
                                                  : SE3());
  for (const auto &frame : preKeyFrame.frames) {
    dataSerializer.process(frame.timestamp);
    dataSerializer.process(preKeyFrame.wasTracked() ? frame.lightBaseToThis
                                                    : AffLight());
  }
}

KeyFrameLoader::KeyFrameLoader(const DatasetReader *datasetReader,
                               const Preprocessor *preprocessor,
                               const fs::path &snapshotDir, CameraBundle *cam,
                               const Settings::KeyFrame &kfSettings,
                               const PointTracerSettings &tracerSettings)
    : datasetReader(datasetReader)
    , preprocessor(preprocessor)
    , snapshotDir(snapshotDir)
    , cam(cam)
    , kfSettings(kfSettings)
    , tracerSettings(tracerSettings) {}

std::unique_ptr<KeyFrame>
KeyFrameLoader::load(const fs::path &keyFrameDir) const {
  int patternSize = tracerSettings.residualPattern.pattern().size();

  std::unique_ptr<PreKeyFrame> preKeyFrame =
      PreKeyFrameLoader(datasetReader, preprocessor, cam, nullptr,
                        keyFrameDir / "pkf.txt", tracerSettings.pyramid)
          .load();
  std::unique_ptr<KeyFrame> keyFrame(
      new KeyFrame(std::move(preKeyFrame), kfSettings, tracerSettings));

  DataSerializer<LOAD> ownData(keyFrameDir / "kf.txt");
  PointSerializer<LOAD> immaturesSerializer(keyFrameDir / "immaturePoints.txt",
                                            patternSize);
  PointSerializer<LOAD> optimizedSerializer(keyFrameDir / "optimizedPoints.txt",
                                            patternSize);

  loadPointVector(ownData, keyFrame->frames[0],
                  keyFrame->frames[0].immaturePoints, immaturesSerializer);
  loadPointVector(ownData, keyFrame->frames[0],
                  keyFrame->frames[0].optimizedPoints, optimizedSerializer);

  ownData.process(keyFrame->frames[0].timestamp);
  ownData.process(keyFrame->thisToWorld);
  ownData.process(keyFrame->frames[0].lightWorldToThis);
  loadTrackedVector(ownData, *keyFrame);

  return keyFrame;
}

template <typename PointT>
void KeyFrameLoader::loadPointVector(
    DataSerializer<LOAD> &ownData, KeyFrameEntry &baseEntry,
    StdVector<PointT> &pointVector,
    PointSerializer<LOAD> &pointSerializer) const {
  int size;
  ownData.process(size);
  pointVector.reserve(size);
  pointVector.clear();
  for (int j = 0; j < size; ++j)
    pointVector.emplace_back(&baseEntry, pointSerializer);
}

void KeyFrameLoader::loadTrackedVector(DataSerializer<LOAD> &ownData,
                                       KeyFrame &keyFrame) const {
  int size;
  ownData.process(size);
  keyFrame.trackedFrames.reserve(size);
  for (int j = 0; j < size; ++j) {
    Timestamp preKeyFrameTs;
    ownData.process(preKeyFrameTs);
    PreKeyFrameLoader preKeyFrameLoader(
        datasetReader, preprocessor, cam, &keyFrame,
        snapshotDir / ("pkf" + std::to_string(preKeyFrameTs) + ".txt"),
        tracerSettings.pyramid);
    keyFrame.trackedFrames.push_back(preKeyFrameLoader.load());
  }
}

KeyFrameSaver::KeyFrameSaver(const fs::path &snapshotDir, int patternSize)
    : snapshotDir(snapshotDir)
    , patternSize(patternSize) {}

void KeyFrameSaver::store(const KeyFrame &keyFrame) const {
  Timestamp ts = keyFrame.preKeyFrame->frames[0].timestamp;
  fs::path keyFrameDir(snapshotDir / ("kf" + std::to_string(ts)));
  fs::create_directories(keyFrameDir);

  DataSerializer<STORE> ownData(keyFrameDir / "kf.txt");

  PreKeyFrameSaver::store(keyFrameDir / "pkf.txt", *keyFrame.preKeyFrame);

  PointSerializer<STORE> immaturePointSerializer(
      keyFrameDir / "immaturePoints.txt", patternSize);
  storePointVector(ownData, keyFrame.frames[0].immaturePoints,
                   immaturePointSerializer);
  PointSerializer<STORE> optimizedPointSerializer(
      keyFrameDir / "optimizedPoints.txt", patternSize);
  storePointVector(ownData, keyFrame.frames[0].optimizedPoints,
                   optimizedPointSerializer);

  ownData.process(ts);
  ownData.process(keyFrame.thisToWorld);
  ownData.process(keyFrame.frames[0].lightWorldToThis);
  storeTrackedVector(ownData, keyFrame);
}

template <typename PointT>
void KeyFrameSaver::storePointVector(
    DataSerializer<STORE> &ownData, const StdVector<PointT> &pointVector,
    PointSerializer<STORE> &pointSerializer) const {
  ownData.process(int(pointVector.size()));
  for (int j = 0; j < pointVector.size(); ++j)
    pointSerializer.process(pointVector[j]);
}

void KeyFrameSaver::storeTrackedVector(DataSerializer<STORE> &ownData,
                                       const KeyFrame &keyFrame) const {
  ownData.process(int(keyFrame.trackedFrames.size()));
  for (int j = 0; j < keyFrame.trackedFrames.size(); ++j) {
    Timestamp preKeyFrameTs = keyFrame.trackedFrames[j]->frames[0].timestamp;
    ownData.process(preKeyFrameTs);
    PreKeyFrameSaver().store(
        snapshotDir / ("pkf" + std::to_string(preKeyFrameTs) + ".txt"),
        *keyFrame.trackedFrames[j]);
  }
}

SnapshotLoader::SnapshotLoader(const DatasetReader *datasetReader,
                               const Preprocessor *preprocessor,
                               CameraBundle *cam, const fs::path &snapshotDir,
                               const Settings &settings)
    : datasetReader(datasetReader)
    , preprocessor(preprocessor)
    , cam(cam)
    , snapshotDir(snapshotDir)
    , settings(settings) {}

void SnapshotLoader::loadDepthColBounds() const {
  fs::path depthCols = snapshotDir / "depth_col.txt";
  std::ifstream ifs(depthCols);
  ifs >> minDepthCol >> maxDepthCol;
}

std::vector<std::unique_ptr<KeyFrame>> SnapshotLoader::load() const {
  CHECK(fs::is_directory(snapshotDir));

  std::vector<std::unique_ptr<KeyFrame>> keyFrames;
  KeyFrameLoader keyFrameLoader(datasetReader, preprocessor, snapshotDir, cam,
                                settings.keyFrame,
                                settings.getPointTracerSettings());
  for (fs::path fname : fs::directory_iterator(snapshotDir)) {
    if (fs::is_directory(fname) && fname.string().size() >= 2 &&
        fname.stem().string().substr(0, 2) == "kf")
      keyFrames.push_back(keyFrameLoader.load(fname));
  }
  std::sort(keyFrames.begin(), keyFrames.end(),
            [](const auto &kf1, const auto &kf2) {
              return kf1->preKeyFrame->frames[0].timestamp <
                     kf2->preKeyFrame->frames[0].timestamp;
            });

  loadDepthColBounds();

  return keyFrames;
}

SnapshotSaver::SnapshotSaver(const fs::path &snapshotDir, int patternSize)
    : snapshotDir(snapshotDir)
    , patternSize(patternSize) {}

void SnapshotSaver::saveDepthColBounds() const {
  fs::path depthCols = snapshotDir / "depth_col.txt";
  std::ofstream ofs(depthCols);
  ofs << minDepthCol << ' ' << maxDepthCol;
}

void SnapshotSaver::save(const KeyFrame *_keyFrames[], int numKeyFrames) const {
  fs::create_directories(snapshotDir);
  KeyFrameSaver keyFrameSaver(snapshotDir, patternSize);
  CHECK(fs::is_directory(snapshotDir));
  for (int j = 0; j < numKeyFrames; ++j)
    keyFrameSaver.store(*_keyFrames[j]);

  saveDepthColBounds();
}

template class PointSerializer<LOAD>;
template class PointSerializer<STORE>;

} // namespace mdso
