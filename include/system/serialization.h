#ifndef INCLUDE_SERIALIZATION
#define INCLUDE_SERIALIZATION

#include "data/DatasetReader.h"
#include "system/ImmaturePoint.h"
#include "system/OptimizedPoint.h"
#include "system/Preprocessor.h"
#include "system/SerializerMode.h"
#include "util/BaseAndTangent.h"
#include "util/types.h"
#include <boost/math/special_functions/nonfinite_num_facets.hpp>
#include <fstream>

namespace mdso {

class PreKeyFrame;
class KeyFrame;
class KeyFrameEntry;
class MarginalizedKeyFrame;

template <SerializerMode mode, typename T> struct RefTHelper;
template <typename T> struct RefTHelper<STORE, T> { using type = const T &; };
template <typename T> struct RefTHelper<LOAD, T> { using type = T &; };

template <SerializerMode mode, typename T>
using RefT = typename RefTHelper<mode, T>::type;

template <SerializerMode mode> class DataSerializer;

template <> class DataSerializer<STORE> {
public:
  DataSerializer(const fs::path &fname);

  template <typename Scalar, int rows, int cols>
  void process(const Eigen::Matrix<Scalar, rows, cols> &mat) {
    for (int i = 0; i < mat.size(); ++i)
      stream << mat.data()[i] << ' ';
    stream << '\n';
  }
  void process(const double &val) { stream << val << '\n'; }
  void process(const int &val) { stream << val << '\n'; }
  void process(const Timestamp &val) { stream << val << '\n'; }
  void process(const AffLight &affLight) {
    stream << affLight.data[0] << ' ' << affLight.data[1] << '\n';
  }
  void process(const SO3 &rot) { process(rot.unit_quaternion().coeffs()); }
  void process(const SE3 &motion) {
    process(motion.so3());
    process(motion.translation());
  }
  void process(const BaseAndTangent<SE3> &baseAndTangent) {
    process(baseAndTangent());
  }
  void process(const ImmaturePoint::State &state) { process(int(state)); }

private:
  std::ofstream stream;
};

template <> class DataSerializer<LOAD> {
public:
  DataSerializer(const fs::path &fname);

  template <typename Scalar, int rows, int cols>
  void process(Eigen::Matrix<Scalar, rows, cols> &mat) {
    for (int i = 0; i < mat.size(); ++i)
      stream >> mat.data()[i];
  }
  void process(double &val) { stream >> val; }
  void process(int &val) { stream >> val; }
  void process(Timestamp &val) { stream >> val; }
  void process(AffLight &affLight) {
    stream >> affLight.data[0] >> affLight.data[1];
  }
  void process(SO3 &rot) {
    Quaternion quaternion;
    process(quaternion.coeffs());
    rot.setQuaternion(quaternion);
  }
  void process(SE3 &motion) {
    process(motion.so3());
    process(motion.translation());
  }
  void process(BaseAndTangent<SE3> &baseAndTangent) {
    SE3 motion;
    process(motion);
    baseAndTangent.setValue(motion);
  }
  void process(ImmaturePoint::State &state) {
    int stateInt;
    process(stateInt);
    state = ImmaturePoint::State(stateInt);
  }

private:
  std::ifstream stream;
};

template <SerializerMode mode> class PointSerializer {
public:
  PointSerializer(const fs::path &pointsFname, int patternSize);

  void process(RefT<mode, ImmaturePoint> p);
  void process(RefT<mode, OptimizedPoint> p);

private:
  DataSerializer<mode> dataSerializer;
  int PS;
};

extern template class PointSerializer<LOAD>;
extern template class PointSerializer<STORE>;

class PreKeyFrameLoader {
public:
  PreKeyFrameLoader(const DatasetReader *datasetReader,
                    const Preprocessor *preprocessor, CameraBundle *cam,
                    KeyFrame *baseFrame, const fs::path &preKeyFrameFname,
                    const Settings::Pyramid &pyramidSettings);
  std::unique_ptr<PreKeyFrame> load() const;

private:
  const DatasetReader *datasetReader;
  const Preprocessor *preprocessor;
  CameraBundle *cam;
  KeyFrame *baseFrame;
  fs::path preKeyFrameFname;
  Settings::Pyramid pyramidSettings;
};

class PreKeyFrameSaver {
public:
  static void store(const fs::path &preKeyFrameFname,
                    const PreKeyFrame &preKeyFrame);
};

class KeyFrameLoader {
public:
  KeyFrameLoader(const DatasetReader *datasetReader,
                 const Preprocessor *preprocessor, const fs::path &snapshotDir,
                 CameraBundle *cam, const Settings::KeyFrame &kfSettings,
                 const PointTracerSettings &tracerSettings);
  std::unique_ptr<KeyFrame> load(const fs::path &keyFrameDir) const;

private:
  template <typename PointT>
  void loadPointVector(DataSerializer<LOAD> &ownData, KeyFrameEntry &baseEntry,
                       StdVector<PointT> &pointVector,
                       PointSerializer<LOAD> &pointSerializer) const;
  void loadTrackedVector(DataSerializer<LOAD> &ownData,
                         KeyFrame &keyFrame) const;

  const DatasetReader *datasetReader;
  const Preprocessor *preprocessor;
  fs::path snapshotDir;
  CameraBundle *cam;
  Settings::KeyFrame kfSettings;
  PointTracerSettings tracerSettings;
};

class KeyFrameSaver {
public:
  KeyFrameSaver(const fs::path &snapshotDir, int patternSize);
  void store(const KeyFrame &keyFrame) const;

private:
  template <typename PointT>
  void storePointVector(DataSerializer<STORE> &ownData,
                        const StdVector<PointT> &pointVector,
                        PointSerializer<STORE> &pointSerializer) const;
  void storeTrackedVector(DataSerializer<STORE> &ownData,
                          const KeyFrame &keyFrame) const;

  fs::path snapshotDir;
  int patternSize;
};

class SnapshotLoader {
public:
  SnapshotLoader(const DatasetReader *datasetReader,
                 const Preprocessor *preprocessor, CameraBundle *cam,
                 const fs::path &snapshotDir, const Settings &settings);
  std::vector<std::unique_ptr<KeyFrame>> load() const;

private:
  void loadDepthColBounds() const;

  const DatasetReader *datasetReader;
  const Preprocessor *preprocessor;
  CameraBundle *cam;
  fs::path snapshotDir;
  Settings settings;
};

class SnapshotSaver {
public:
  SnapshotSaver(const fs::path &snapshotDir, int patternSize);

  void save(const KeyFrame *keyFrames[], int numKeyFrames) const;

private:
  void saveDepthColBounds() const;

  fs::path snapshotDir;
  int patternSize;
};

} // namespace mdso

#endif
