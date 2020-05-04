#ifndef INCLUDE_ENERGYFUNCTIONHESSIAN
#define INCLUDE_ENERGYFUNCTIONHESSIAN

#include "optimize/Accumulator.h"
#include "optimize/Gradient.h"
#include "optimize/Parameters.h"
#include "optimize/Residual.h"
#include "util/types.h"

namespace mdso::optimize {

class Hessian {
public:
  using MatFFt = MatXXt;
  using MatFPt = MatXXt;
  using VecPt = VecXt;

  class AccumulatedBlocks {
  public:
    AccumulatedBlocks(int numKeyFrames, int numCameras, int numPoints);

    int numKeyFrames() const;
    int numCameras() const;
    int numPoints() const;

    void add(const Residual &residual,
             const Residual::DeltaHessian &deltaHessian);

    const Accumulator<Mat77t> &motionMotion(int frameInd1, int frameInd2) const;
    const Accumulator<Mat72t> &motionAff(int frameInd1, int frameInd2,
                                         int frameCamInd2) const;
    const Accumulator<Mat22t> &affAff(int frameInd1, int frameCamInd1,
                                      int frameInd2, int frameCamInd2) const;
    const Accumulator<Vec7t> &motionPoint(int frameInd, int pointInd) const;
    const Accumulator<Vec2t> &affPoint(int frameInd, int frameCamInd,
                                       int pointInd) const;
    const Accumulator<T> &pointPoint(int pointInd) const;

  private:
    Accumulator<Mat77t> &motionMotion(int frameInd1, int frameInd2);
    Accumulator<Mat72t> &motionAff(int frameInd1, int frameInd2,
                                   int frameCamInd2);
    Accumulator<Mat22t> &affAff(int frameInd1, int frameCamInd1, int frameInd2,
                                int frameCamInd2);
    Accumulator<Vec7t> &motionPoint(int frameInd, int pointInd);
    Accumulator<Vec2t> &affPoint(int frameInd, int frameCamInd, int pointInd);
    Accumulator<T> &pointPoint(int pointInd);

    void add(const Residual::FrameFrameHessian &frameFrameHessian, int f1i,
             int f1ci, int f2i, int f2ci);
    void add(const Residual::FramePointHessian &framePointHessian, int fi,
             int fci, int pi);

    Array2d<Accumulator<Mat77t>> mMotionMotion;
    Array3d<Accumulator<Mat72t>> mMotionAff;
    Array4d<Accumulator<Mat22t>> mAffAff;
    Array2d<Accumulator<Vec7t>> mMotionPoint;
    Array3d<Accumulator<Vec2t>> mAffPoint;

    std::vector<Accumulator<T>> mPointPoint;
  };

  Hessian(const AccumulatedBlocks &accumulatedBlocks,
          const Parameters::Jacobians &parameterJacobians,
          const Settings::Optimization &settings);

  Hessian levenbergMarquardtDamp(double lambda) const;
  DeltaParameterVector solve(const Gradient &gradient,
                             const int *excluedPointInds = nullptr,
                             int excludedPointIndsSize = 0) const;
  T applyQuadraticForm(const DeltaParameterVector &delta) const;

  inline MatFFt getFrameFrame() const { return frameFrame; }
  inline MatFPt getFramePoint() const { return framePoint; }
  inline VecPt getPointPoint() const { return pointPoint; }

private:
  inline Eigen::Block<MatFFt, sndDoF, sndDoF> sndSndBlock() {
    return frameFrame.block<sndDoF, sndDoF>(
        frameParameterOrder.frameToWorld(1),
        frameParameterOrder.frameToWorld(1));
  }

  inline Eigen::Block<MatFFt, sndDoF, restDoF> sndRestBlock(int restFrameInd) {
    CHECK_GE(restFrameInd, 1);
    return frameFrame.block<sndDoF, restDoF>(
        frameParameterOrder.frameToWorld(1),
        frameParameterOrder.frameToWorld(restFrameInd));
  }

  inline Eigen::Block<MatFFt, restDoF, restDoF> restRestBlock(int frameInd1,
                                                              int frameInd2) {
    CHECK_GE(frameInd1, 1);
    CHECK_GE(frameInd2, 1);
    return frameFrame.block<restDoF, restDoF>(
        frameParameterOrder.frameToWorld(frameInd1),
        frameParameterOrder.frameToWorld(frameInd2));
  }

  inline Eigen::Block<MatFFt, sndDoF, affDoF> sndAffBlock(int frameInd,
                                                          int camInd) {
    return frameFrame.block<sndDoF, affDoF>(
        frameParameterOrder.frameToWorld(1),
        frameParameterOrder.lightWorldToFrame(frameInd, camInd));
  }

  inline Eigen::Block<MatFFt, restDoF, affDoF>
  restAffBlock(int frame1Ind, int frame2Ind, int cam2Ind) {
    return frameFrame.block<restDoF, affDoF>(
        frameParameterOrder.frameToWorld(frame1Ind),
        frameParameterOrder.lightWorldToFrame(frame2Ind, cam2Ind));
  }

  inline Eigen::Block<MatFFt, affDoF, affDoF>
  affAffBlock(int frame1Ind, int cam1Ind, int frame2Ind, int cam2Ind) {
    return frameFrame.block<affDoF, affDoF>(
        frameParameterOrder.lightWorldToFrame(frame1Ind, cam1Ind),
        frameParameterOrder.lightWorldToFrame(frame2Ind, cam2Ind));
  }

  inline Eigen::Block<MatFPt, sndDoF, pointDoF> sndPointBlock(int pointInd) {
    return framePoint.block<sndDoF, pointDoF>(
        frameParameterOrder.frameToWorld(1), pointInd);
  }

  inline Eigen::Block<MatFPt, restDoF, pointDoF> restPointBlock(int frameInd,
                                                                int pointInd) {
    return framePoint.block<restDoF, pointDoF>(
        frameParameterOrder.frameToWorld(frameInd), pointInd);
  }

  inline Eigen::Block<MatFPt, affDoF, pointDoF>
  affPointBlock(int frameInd, int camInd, int pointInd) {
    return framePoint.block<affDoF, pointDoF>(
        frameParameterOrder.lightWorldToFrame(frameInd, camInd), pointInd);
  }

  inline T &pointPointBlock(int pointInd) {
    CHECK_GE(pointInd, 0);
    CHECK_LT(pointInd, pointPoint.size());

    return pointPoint[pointInd];
  }

  void fillLowerBlocks();

  FrameParameterOrder frameParameterOrder;
  MatFFt frameFrame;
  MatFPt framePoint;
  VecPt pointPoint;
  Settings::Optimization settings;
};

} // namespace mdso::optimize

#endif
