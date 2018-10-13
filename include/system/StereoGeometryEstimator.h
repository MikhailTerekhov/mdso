#include "system/CameraModel.h"

namespace fishdso {

class StereoGeometryEstimator {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  StereoGeometryEstimator(
      CameraModel *cam, const StdVector<std::pair<Vec2, Vec2>> &imgCorresps);

  SE3 findCoarseMotion();
  SE3 findPreciseMotion();

  const std::vector<int> &inliersInds() const;
  const std::vector<std::pair<double, double>> &depths();
  int inliersNum();

private:
  struct ReprojectionResidual {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ReprojectionResidual(CameraModel *cam, const Vec3 &p, const Vec3 &q,
                         const Vec2 &pMapped, const Vec2 &qMapped)
        : cam(cam), p(p), q(q), pMapped(pMapped), qMapped(qMapped) {}

    template <typename T>
    bool operator()(const T *const rotP, const T *const transP, T *res) const {
      typedef Eigen::Matrix<T, 2, 1> Vec2t;
      typedef Eigen::Matrix<T, 3, 1> Vec3t;
      typedef Eigen::Matrix<T, 3, 3> Mat33t;
      typedef Eigen::Quaternion<T> Quatt;
      typedef Sophus::SO3<T> SO3t;

      Eigen::Map<const Vec3t> transM(transP);
      Vec3t trans(transM);
      Eigen::Map<const Quatt> rotM(rotP);
      Quatt rot(rotM);

      Vec3t pT = p.cast<T>(), qT = q.cast<T>();
      Vec2t pMappedT = pMapped.cast<T>(), qMappedT = qMapped.cast<T>();

      Mat33t E = SO3t::hat(trans) * rot.toRotationMatrix();
      Vec3t Ep = E * pT, Etq = E.transpose() * qT;

      T EpSqN = Ep.squaredNorm(), EtqSqN = Etq.squaredNorm();
      if (EpSqN < T(1e-4) || EtqSqN < T(1e-4)) {
        res[0] = T(0.0);
        res[1] = T(0.0);
        res[2] = T(0.0);
        res[3] = T(0.0);
        return true;
      }

      Vec3t pProj = pT - (pT.dot(Etq) / EtqSqN) * Etq;
      Vec3t qProj = qT - (qT.dot(Ep) / EpSqN) * Ep;

      Vec2t errVec1 = cam->map(pProj.data()) - pMappedT;
      Vec2t errVec2 = cam->map(qProj.data()) - qMappedT;

      res[0] = errVec1[0];
      res[1] = errVec1[1];
      res[2] = errVec2[0];
      res[3] = errVec2[1];

      return true;
    }

  private:
    CameraModel *cam;
    Vec3 p, q;
    Vec2 pMapped, qMapped;
  };

  static EIGEN_STRONG_INLINE std::vector<int> reservedVector() {
    std::vector<int> res;
    res.reserve(settingKeyPointsCount);
    return res;
  }

  int findInliersEssential(const Mat33 &E, std::vector<int> &_inliersInds);
  int findInliersMotion(const SE3 &motion, std::vector<int> &_inliersInds);

  SE3 extractMotion(const Mat33 &E, std::vector<int> &_inliersInds,
                    int &newInliers);

  CameraModel *cam;
  StdVector<std::pair<Vec2, Vec2>> imgCorresps;
  std::vector<std::pair<Vec3, Vec3>> rays;
  std::vector<std::pair<double, double>> _depths;
  std::vector<int> _inliersInds;
  SE3 motion;
  bool coarseFound, preciseFound, depthsEvaluated;
};

} // namespace fishdso
