#include "system/serialization.h"
#include "system/KeyFrame.h"
#include <boost/math/special_functions/nonfinite_num_facets.hpp>

namespace fishdso {

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
  dataSerializer.process(
      p.baseDirections[0]); // for compatibility with multicamera version
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
  Vec3 dir = Vec3::Zero();
  dataSerializer.process(dir); // for compatibility with multicamera version

  if constexpr (mode == LOAD) {
    double logDepth = 1;
    dataSerializer.process(logDepth);
    p.logInvDepth = -logDepth;
    dataSerializer.process(p.stddev);
    double minDepth, maxDepth;
    dataSerializer.process(minDepth);
    dataSerializer.process(maxDepth);
  } else {
    double logDepth = -p.logInvDepth;
    dataSerializer.process(logDepth);
    dataSerializer.process(p.stddev);
    double minDepth = p.depth(), maxDepth = p.depth();
    dataSerializer.process(minDepth);
    dataSerializer.process(maxDepth);
  }
}

PreKeyFrameLoader::PreKeyFrameLoader(const MultiFovReader *datasetReader,
                                     CameraModel *cam, KeyFrame *baseFrame,
                                     const fs::path &preKeyFrameFname,
                                     const Settings::Pyramid &pyramidSettings)
    : datasetReader(datasetReader)
    , cam(cam)
    , baseFrame(baseFrame)
    , preKeyFrameFname(preKeyFrameFname)
    , pyramidSettings(pyramidSettings) {}

std::shared_ptr<PreKeyFrame> PreKeyFrameLoader::load() const {
  DataSerializer<LOAD> dataSerializer(preKeyFrameFname);

  SE3 baseToTracked;
  AffLight lightBaseToTracked;
  int globalFrameNum;
  dataSerializer.process(baseToTracked);
  dataSerializer.process(globalFrameNum);
  dataSerializer.process(lightBaseToTracked);

  cv::Mat frame = datasetReader->getFrame(globalFrameNum);

  std::shared_ptr<PreKeyFrame> preKeyFrame(
      new PreKeyFrame(baseFrame, cam, frame, globalFrameNum, pyramidSettings));
  preKeyFrame->baseToThis = baseToTracked;
  preKeyFrame->lightBaseToThis = lightBaseToTracked;
  return preKeyFrame;
}

void PreKeyFrameSaver::store(const fs::path &preKeyFrameFname,
                             const PreKeyFrame &preKeyFrame) {
  DataSerializer<STORE> dataSerializer(preKeyFrameFname);
  dataSerializer.process(preKeyFrame.baseToThis);
  dataSerializer.process(preKeyFrame.globalFrameNum);
  dataSerializer.process(preKeyFrame.lightBaseToThis);
}

KeyFrameLoader::KeyFrameLoader(const MultiFovReader *datasetReader,
                               const fs::path &snapshotDir, CameraModel *cam,
                               const Settings::KeyFrame &kfSettings,
                               const PointTracerSettings &tracerSettings)
    : datasetReader(datasetReader)
    , snapshotDir(snapshotDir)
    , cam(cam)
    , kfSettings(kfSettings)
    , tracerSettings(tracerSettings) {}

void KeyFrameLoader::load(const fs::path &keyFrameDir,
                          StdMap<int, KeyFrame> &keyFrames) const {
  int patternSize = tracerSettings.residualPattern.pattern().size();

  std::shared_ptr<PreKeyFrame> preKeyFrame =
      PreKeyFrameLoader(datasetReader, cam, nullptr, keyFrameDir / "pkf.txt",
                        tracerSettings.pyramid)
          .load();
  int frameNum = preKeyFrame->globalFrameNum;
  auto [keyFrameIt, insertionOk] = keyFrames.insert(
      {frameNum, KeyFrame(std::move(preKeyFrame), kfSettings, tracerSettings)});
  CHECK(insertionOk);
  KeyFrame &keyFrame = keyFrameIt->second;

  DataSerializer<LOAD> ownData(keyFrameDir / "kf.txt");
  PointSerializer<LOAD> immaturesSerializer(keyFrameDir / "immaturePoints.txt",
                                            patternSize);
  PointSerializer<LOAD> optimizedSerializer(keyFrameDir / "optimizedPoints.txt",
                                            patternSize);

  loadPointVector(ownData, keyFrame, keyFrame.immaturePoints,
                  immaturesSerializer);
  loadPointVector(ownData, keyFrame, keyFrame.optimizedPoints,
                  optimizedSerializer);

  int globalFrameNum;
  ownData.process(globalFrameNum);
  CHECK_EQ(globalFrameNum, keyFrame.preKeyFrame->globalFrameNum);
  ownData.process(keyFrame.thisToWorld);
  ownData.process(keyFrame.lightWorldToThis);
  loadTrackedVector(ownData, keyFrame);
}

template <typename PointT>
void KeyFrameLoader::loadPointVector(
    DataSerializer<LOAD> &ownData, KeyFrame &baseFrame,
    std::vector<std::unique_ptr<PointT>> &pointVector,
    PointSerializer<LOAD> &pointSerializer) const {
  int size;
  ownData.process(size);
  pointVector.reserve(size);
  pointVector.clear();
  for (int j = 0; j < size; ++j)
    pointVector.emplace_back(new PointT(&baseFrame, pointSerializer));
}

void KeyFrameLoader::loadTrackedVector(DataSerializer<LOAD> &ownData,
                                       KeyFrame &keyFrame) const {
  int size;
  ownData.process(size);
  keyFrame.trackedFrames.reserve(size);
  for (int j = 0; j < size; ++j) {
    int preKeyFrameNum;
    ownData.process(preKeyFrameNum);
    PreKeyFrameLoader preKeyFrameLoader(
        datasetReader, cam, &keyFrame,
        snapshotDir / ("pkf" + std::to_string(preKeyFrameNum) + ".txt"),
        tracerSettings.pyramid);
    keyFrame.trackedFrames.push_back(preKeyFrameLoader.load());
  }
}

KeyFrameSaver::KeyFrameSaver(const fs::path &snapshotDir, int patternSize)
    : snapshotDir(snapshotDir)
    , patternSize(patternSize) {}

void KeyFrameSaver::store(const KeyFrame &keyFrame) const {
  int frameNum = keyFrame.preKeyFrame->globalFrameNum;
  fs::path keyFrameDir(snapshotDir / ("kf" + std::to_string(frameNum)));
  fs::create_directories(keyFrameDir);

  DataSerializer<STORE> ownData(keyFrameDir / "kf.txt");

  PreKeyFrameSaver::store(keyFrameDir / "pkf.txt", *keyFrame.preKeyFrame);

  PointSerializer<STORE> immaturePointSerializer(
      keyFrameDir / "immaturePoints.txt", patternSize);
  storePointVector(ownData, keyFrame.immaturePoints, immaturePointSerializer);
  PointSerializer<STORE> optimizedPointSerializer(
      keyFrameDir / "optimizedPoints.txt", patternSize);
  storePointVector(ownData, keyFrame.optimizedPoints, optimizedPointSerializer);

  ownData.process(frameNum);
  ownData.process(keyFrame.thisToWorld);
  ownData.process(keyFrame.lightWorldToThis);
  storeTrackedVector(ownData, keyFrame);
}

template <typename PointT>
void KeyFrameSaver::storePointVector(
    DataSerializer<STORE> &ownData,
    const std::vector<std::unique_ptr<PointT>> &pointVector,
    PointSerializer<STORE> &pointSerializer) const {
  ownData.process(int(pointVector.size()));
  for (int j = 0; j < pointVector.size(); ++j)
    pointSerializer.process(*pointVector[j]);
}

void KeyFrameSaver::storeTrackedVector(DataSerializer<STORE> &ownData,
                                       const KeyFrame &keyFrame) const {
  ownData.process(int(keyFrame.trackedFrames.size()));
  for (int j = 0; j < keyFrame.trackedFrames.size(); ++j) {
    int preKeyFrameNum = keyFrame.trackedFrames[j]->globalFrameNum;
    ownData.process(preKeyFrameNum);
    PreKeyFrameSaver().store(
        snapshotDir / ("pkf" + std::to_string(preKeyFrameNum) + ".txt"),
        *keyFrame.trackedFrames[j]);
  }
}

SnapshotLoader::SnapshotLoader(const MultiFovReader *datasetReader,
                               CameraModel *cam, const fs::path &snapshotDir,
                               const Settings &settings)
    : datasetReader(datasetReader)
    , cam(cam)
    , snapshotDir(snapshotDir)
    , settings(settings) {}

void SnapshotLoader::load(StdMap<int, KeyFrame> &keyFrames) const {
  CHECK(fs::is_directory(snapshotDir));

  KeyFrameLoader keyFrameLoader(datasetReader, snapshotDir, cam,
                                settings.keyFrame,
                                settings.getPointTracerSettings());
  for (fs::path fname : fs::directory_iterator(snapshotDir)) {
    if (fs::is_directory(fname) && fname.string().size() >= 2 &&
        fname.stem().string().substr(0, 2) == "kf")
      keyFrameLoader.load(fname, keyFrames);
  }
}

SnapshotSaver::SnapshotSaver(const fs::path &snapshotDir, int patternSize)
    : snapshotDir(snapshotDir)
    , patternSize(patternSize) {}

void SnapshotSaver::save(const KeyFrame *_keyFrames[], int numKeyFrames) const {
  fs::create_directories(snapshotDir);
  KeyFrameSaver keyFrameSaver(snapshotDir, patternSize);
  CHECK(fs::is_directory(snapshotDir));
  for (int j = 0; j < numKeyFrames; ++j)
    keyFrameSaver.store(*_keyFrames[j]);
}

template class PointSerializer<LOAD>;
template class PointSerializer<STORE>;

} // namespace fishdso
