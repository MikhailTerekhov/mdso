#include "system/CameraModel.h"
#include <util/ObjectPool.h>

namespace fishdso {

class StereoGeometryEstimator {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  StereoGeometryEstimator(CameraModel *cam,
                          const StdVector<std::pair<Vec2, Vec2>> &imgCorresps);

  SE3 findCoarseMotion();
  SE3 findPreciseMotion();

  const std::vector<int> &inliersInds() const;
  const std::vector<std::pair<double, double>> &depths();
  int inliersNum();

  void outputInlierCorresps();

private:
  int findInliersEssential(const Mat33 &E, std::vector<int> &_inliersInds);
  int findInliersMotion(const SE3 &motion, std::vector<int> &_inliersInds);

  SE3 extractMotion(const Mat33 &E, std::vector<int> &_inliersInds,
                    int &newInliers, bool doLogFrontPoints);

  CameraModel *cam;
  StdVector<std::pair<Vec2, Vec2>> imgCorresps;
  std::vector<std::pair<Vec3, Vec3>> rays;
  std::vector<std::pair<double, double>> _depths;

  std::vector<int> _inliersInds;
  ObjectPool<std::vector<int>> inlierVectorsPool;

  SE3 motion;
  bool coarseFound, preciseFound, depthsEvaluated;
};

} // namespace fishdso
