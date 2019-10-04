#ifndef INCLUDE_TRIANGULATION
#define INCLUDE_TRIANGULATION

#include "system/CameraModel.h"
#include "util/types.h"
#include <opencv2/opencv.hpp>
#include <unordered_set>
#include <vector>

namespace mdso {

class Triangulation {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  struct Vertex;
  struct Edge;
  struct Triangle;

  typedef std::vector<const Vertex *>::const_iterator VertexIterator;

  struct Vertex {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Vec2 pos;
    std::unordered_set<Edge *> edges;
    std::unordered_set<Triangle *> triangles;
    int index;
  };

  struct Edge {
    Vertex *vert[2];
    Triangle *triang[2];
  };

  struct Triangle {
    Vertex *vert[3];
    Edge *edges[3];
  };

  static const int POINT_NOT_FOUND = -4;

  Triangulation(const StdVector<Vec2> &newPoints,
                const Settings::Triangulation &settings = {});

  VertexIterator begin() const;
  VertexIterator end() const;

  Vertex *operator[](int index) const;
  const std::vector<const Edge *> &edges() const;
  const std::vector<const Triangle *> &triangles() const;

  Triangle *enclosingTriangle(const Vec2 &point);
  bool isIncidentToBoundary(const Triangle *tri) const;

  cv::Mat draw(int imgWidth, int imgHeight, cv::Scalar bgCol,
               cv::Scalar edgeCol) const;
  void draw(cv::Mat &img, cv::Scalar edgeCol) const;
  void draw(cv::Mat &img, cv::Scalar edgeCol, int thickness) const;

  void drawCurved(CameraModel *cam, cv::Mat &img, cv::Scalar edgeCol) const;

private:
  inline static Vertex *findThirdVert(Triangle *tri, Edge *edge);

  inline static Edge *findOppositeEdge(Triangle *tri, Vertex *vert);

  inline static bool isIncident(Vertex *vert, Edge *edge);
  inline static bool isIncident(Vertex *vert, Triangle *tri);

  inline bool isInsideBound(const Vec2 &point) const;

  inline bool doesContain(Triangle *tri, const Vec2 &point) const;
  inline bool doesContain(Edge *edge, const Vec2 &point) const;

  inline bool isFromBoundingTri(Vertex *vert) const;
  inline bool isEdgeLegal(Edge *edge) const;

  Vertex *makeBoundingVertex(const Vec2 &newPos);
  Vertex *makeVertex(const Vec2 &newPos, int newIndex);
  Edge *makeEdge();
  Triangle *makeTriangle();

  void performFlip(Edge *edge);
  void divideTriangle(Triangle *tri, Vertex *vert);
  void divideEdge(Edge *edge, Vertex *vert);

  Vertex *addPoint(Vertex *newVert);

  void tidy();

  void fillPtrVectors();

  void drawScaled(cv::Mat &img, double scaleX, double scaleY,
                  cv::Point upperLeftPoint, cv::Scalar edgeCol) const;
  void drawScaled(cv::Mat &img, double scaleX, double scaleY,
                  cv::Point upperLeftPoint, cv::Scalar edgeCol,
                  int thickness) const;

  double maxDim;
  Vec2 upperLeft, bottomRight;
  std::vector<std::unique_ptr<Vertex>> _vertices;
  std::vector<std::unique_ptr<Edge>> _edges;
  std::vector<std::unique_ptr<Triangle>> _triangles;
  std::vector<int> indicesInv;

  std::vector<const Vertex *> vertexPtrs;
  std::vector<const Edge *> edgePtrs;
  std::vector<const Triangle *> trianglePtrs;

  //  std::vector<Edge *> _edges;
  //  std::vector<Triangle *> _triangles;

  std::mt19937 mt;

  Settings::Triangulation settings;
};

void drawCurvedInternal(CameraModel *cam, Vec2 ptFrom, Vec2 ptTo, cv::Mat &img,
                        cv::Scalar edgeCol);

} // namespace mdso

#endif
