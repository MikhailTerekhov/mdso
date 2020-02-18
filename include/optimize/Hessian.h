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
    friend class Hessian;

  public:
    AccumulatedBlocks(int numKeyFrames, int numCameras, int numPoints);

    int numKeyFrames() const;
    int numCameras() const;
    int numPoints() const;

    void add(const Residual &residual,
             const Residual::DeltaHessian &deltaHessian);

  private:
    void add(const Residual::FrameFrameHessian &frameFrameHessian, int f1i,
             int f1ci, int f2i, int f2ci);
    void add(const Residual::FramePointHessian &framePointHessian, int fi,
             int fci, int pi);

    Array2d<Accumulator<Mat77t>> motionMotion;
    Array3d<Accumulator<Mat72t>> motionAff;
    Array4d<Accumulator<Mat22t>> affAff;
    Array2d<Accumulator<Vec7t>> motionPoint;
    Array3d<Accumulator<Vec2t>> affPoint;

    std::vector<Accumulator<T>> pointPoint;
  };

  Hessian(const AccumulatedBlocks &accumulatedBlocks,
          const Parameters::Jacobians &parameterJacobians,
          const Settings::Optimization &settings);

  Hessian levenbergMarquardtDamp(double lambda) const;
  DeltaParameterVector solve(const Gradient &gradient) const;

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
