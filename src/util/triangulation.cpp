#include "util/triangulation.h"
#include "util/defs.h"
#include "util/types.h"
#include "util/util.h"
#include <opencv2/opencv.hpp>
#include <queue>

namespace fishdso {

Triangulation::Triangulation(const std::vector<Vec2> &newPoints)
    : indicesInv(newPoints.size()) {
  _vertices.reserve(newPoints.size() + 3);

  double minx = std::numeric_limits<double>::infinity(),
         miny = std::numeric_limits<double>::infinity(),
         maxx = -std::numeric_limits<double>::infinity(),
         maxy = -std::numeric_limits<double>::infinity();

  for (Vec2 p : newPoints) {
    if (p[0] < minx)
      minx = p[0];
    if (p[1] < miny)
      miny = p[1];
    if (p[0] > maxx)
      maxx = p[0];
    if (p[1] > maxy)
      maxy = p[1];
  }

  maxDim = std::max(maxx - minx, maxy - miny);
  upperLeft = Vec2(minx, miny);
  bottomRight = Vec2(maxx, maxy);

  makeBoundingVertex(Vec2(minx - maxDim, miny - maxDim));
  makeBoundingVertex(Vec2(minx + 4 * maxDim, miny - maxDim));
  makeBoundingVertex(Vec2(minx - maxDim, miny + 4 * maxDim));

  Triangle *tri = makeTriangle();
  Edge *edges[] = {makeEdge(), makeEdge(), makeEdge()};
  for (int edgesInd = 0; edgesInd < 3; ++edgesInd) {
    const auto &curEdge = edges[edgesInd];
    curEdge->vert[0] = _vertices[edgesInd].get();
    _vertices[edgesInd]->edges.insert(curEdge);
    curEdge->vert[1] = _vertices[(edgesInd + 1) % 3].get();
    _vertices[(edgesInd + 1) % 3]->edges.insert(curEdge);
    curEdge->triang[0] = tri;
    curEdge->triang[1] = nullptr;

    tri->edges[edgesInd] = curEdge;
  }

  for (int pointsInd = 0; pointsInd < 3; ++pointsInd) {
    _vertices[pointsInd]->triangles.insert(tri);
    tri->vert[pointsInd] = _vertices[pointsInd].get();
  }

  std::vector<int> indices(newPoints.size());
  for (int i = 0; i < int(indices.size()); ++i)
    indices[i] = i;
  std::shuffle(indices.begin(), indices.end(), mt);

  for (int i = 0; i < int(indices.size()); ++i) {
    Vertex *vert = makeVertex(newPoints[indices[i]], indices[i]);
    Vertex *vertCreated = addPoint(vert);
    if (vertCreated != vert) {
      // std::cout << "smth repeated!" << std::endl;
      indicesInv[indices[i]] = indicesInv[vertCreated->index];
      _vertices.pop_back();
    } else
      indicesInv[indices[i]] = i + 3;
  }

  tidy();
  fillPtrVectors();
}

Triangulation::VertexIterator Triangulation::begin() const {
  return vertexPtrs.begin();
}

Triangulation::VertexIterator Triangulation::end() const {
  return vertexPtrs.end();
}

Triangulation::Vertex *Triangulation::operator[](int index) const {
  if (index < 0 || index >= int(_vertices.size()) - 3)
    throw std::runtime_error("Triangulation: index out of bounds");
  return _vertices[indicesInv[index]].get();
}

const std::vector<const Triangulation::Edge *> &Triangulation::edges() const {
  return edgePtrs;
}

const std::vector<const Triangulation::Triangle *> &
Triangulation::triangles() const {
  return trianglePtrs;
}

Triangulation::Vertex *Triangulation::makeBoundingVertex(const Vec2 &newPos) {
  _vertices.push_back(std::unique_ptr<Vertex>(new Vertex()));
  _vertices.back()->pos = newPos;
  _vertices.back()->index = -1;

  return _vertices.back().get();
}

Triangulation::Vertex *Triangulation::makeVertex(const Vec2 &newPos,
                                                 int newIndex) {
  _vertices.push_back(std::unique_ptr<Vertex>(new Vertex()));
  _vertices.back()->pos = newPos;
  _vertices.back()->index = newIndex;

  return _vertices.back().get();
}

Triangulation::Edge *Triangulation::makeEdge() {
  _edges.push_back(std::unique_ptr<Edge>(new Edge()));
  return _edges.back().get();
}

Triangulation::Triangle *Triangulation::makeTriangle() {
  _triangles.push_back(std::unique_ptr<Triangle>(new Triangle()));
  return _triangles.back().get();
}

void Triangulation::performFlip(Edge *oldEdge) {
  // inc comes from "incident"
  Vertex *incVert[] = {oldEdge->vert[0], oldEdge->vert[1]};
  Triangle *oldTri[] = {oldEdge->triang[0], oldEdge->triang[1]};
  Vertex *apartVert[] = {findThirdVert(oldTri[0], oldEdge),
                         findThirdVert(oldTri[1], oldEdge)};

  Edge *newEdge = makeEdge();
  Triangle *incTri[] = {makeTriangle(), makeTriangle()};

  for (int apartVertInd = 0; apartVertInd < 2; ++apartVertInd)
    newEdge->vert[apartVertInd] = apartVert[apartVertInd];
  for (int triIncInd = 0; triIncInd < 2; ++triIncInd)
    newEdge->triang[triIncInd] = incTri[triIncInd];

  for (int oldTriInd = 0; oldTriInd < 2; ++oldTriInd) {
    const auto &curOldTri = oldTri[oldTriInd];
    for (const auto &curEdge : curOldTri->edges) {
      if (curEdge == oldEdge)
        continue;
      int edgeToTriInd = curEdge->triang[0] == curOldTri ? 0 : 1;
      int incVertInd = isIncident(incVert[0], curEdge) ? 0 : 1;
      curEdge->triang[edgeToTriInd] = incTri[incVertInd];
      incTri[incVertInd]->edges[oldTriInd] = curEdge;
    }
  }

  for (const auto &curIncTri : incTri)
    curIncTri->edges[2] = newEdge;

  for (const auto &curIncTri : incTri)
    for (int apartVertInd = 0; apartVertInd < 2; ++apartVertInd)
      curIncTri->vert[apartVertInd] = apartVert[apartVertInd];
  for (int incVertInd = 0; incVertInd < 2; ++incVertInd)
    incTri[incVertInd]->vert[2] = incVert[incVertInd];

  for (const auto &curIncVert : incVert) {
    curIncVert->edges.erase(oldEdge);
    for (const auto &curOldTri : oldTri)
      curIncVert->triangles.erase(curOldTri);
  }

  for (int apartVertInd = 0; apartVertInd < 2; ++apartVertInd)
    apartVert[apartVertInd]->triangles.erase(oldTri[apartVertInd]);

  for (int incVertInd = 0; incVertInd < 2; ++incVertInd) {
    const auto &curIncVert = incVert[incVertInd];
    curIncVert->edges.erase(oldEdge);
    for (const auto &curOldTri : oldTri)
      curIncVert->triangles.erase(curOldTri);
    curIncVert->triangles.insert(incTri[incVertInd]);
  }

  for (const auto &curApartVert : apartVert) {
    curApartVert->edges.insert(newEdge);
    for (const auto &curIncTri : incTri)
      curApartVert->triangles.insert(curIncTri);
  }
}

EIGEN_STRONG_INLINE Triangulation::Vertex *
Triangulation::findThirdVert(Triangle *tri, Edge *edge) {
  bool edgeFound = false;
  for (int i = 0; i < 3; ++i) {
    if (tri->edges[i] == edge) {
      edgeFound = true;
      break;
    }
  }
  if (!edgeFound)
    throw std::runtime_error("triangulation inner problem: bad findThirdVert");

  for (int i = 0; i < 3; ++i)
    if (tri->vert[i] != edge->vert[0] && tri->vert[i] != edge->vert[1])
      return tri->vert[i];
}

EIGEN_STRONG_INLINE Triangulation::Edge *
Triangulation::findOppositeEdge(Triangulation::Triangle *tri,
                                Triangulation::Vertex *vert) {
  bool vertFound = false;
  for (const auto &triVert : tri->vert) {
    if (triVert == vert) {
      vertFound = true;
      break;
    }
  }
  if (!vertFound)
    throw std::runtime_error(
        "triangulation inner problem: bad findOppositeEdge");

  for (const auto &triSide : tri->edges)
    if (triSide->vert[0] != vert && triSide->vert[1] != vert)
      return triSide;
}

EIGEN_STRONG_INLINE bool Triangulation::isIncident(Vertex *vert, Edge *edge) {
  return edge->vert[0] == vert || edge->vert[1] == vert;
}

EIGEN_STRONG_INLINE bool Triangulation::isIncident(Vertex *vert,
                                                   Triangle *tri) {
  return tri->vert[0] == vert || tri->vert[1] == vert || tri->vert[2] == vert;
}

EIGEN_STRONG_INLINE double cross2(const Vec2 &a, const Vec2 &b) {
  return a[0] * b[1] - a[1] * b[0];
}

EIGEN_STRONG_INLINE bool isSameSide(const Vec2 &a, const Vec2 &b,
                                    const Vec2 &p1, const Vec2 &p2) {
  return cross2(b - a, p1 - a) * cross2(b - a, p2 - a) >= 0;
}

EIGEN_STRONG_INLINE bool isInsideTriangle(const Vec2 &a, const Vec2 &b,
                                          const Vec2 &c, const Vec2 &p) {
  return isSameSide(a, b, p, c) && isSameSide(a, c, p, b) &&
         isSameSide(b, c, p, a);
}

EIGEN_STRONG_INLINE bool isABCDConvex(const Vec2 &a, const Vec2 &b,
                                      const Vec2 &c, const Vec2 &d) {
  return !(isInsideTriangle(b, c, d, a) || isInsideTriangle(a, c, d, b) ||
           isInsideTriangle(a, b, d, c) || isInsideTriangle(a, b, c, d));
}

EIGEN_STRONG_INLINE bool isABLegal(const Vec2 &a, const Vec2 &b, const Vec2 &c,
                                   const Vec2 &d) {
  Vec2 da = a - d;
  Vec2 db = b - d;
  Vec2 dc = c - d;
  Vec2 ab = b - a;
  Vec2 bc = c - b;
  Mat33 checker;
  // clang-format off
  checker << da[0], da[1], da.squaredNorm(),
             db[0], db[1], db.squaredNorm(),
             dc[0], dc[1], dc.squaredNorm();
  // clang-format on

  return checker.determinant() * cross2(ab, bc) <= 0;
}

EIGEN_STRONG_INLINE bool areEqual(const Vec2 &a, const Vec2 &b, double eps) {
  return (a - b).norm() < eps;
}

EIGEN_STRONG_INLINE bool doesABcontain(const Vec2 &a, const Vec2 &b,
                                       const Vec2 &p, double eps) {
  Vec2 ap = p - a;
  double abNorm = (b - a).norm();
  Vec2 abN = (b - a) / abNorm;
  double abNp = abN.dot(ap);
  return std::abs(ap.dot(Vec2(abN[1], -abN[0]))) < eps && abNp >= -eps &&
         abNp <= abNorm + eps;
}

EIGEN_STRONG_INLINE bool
Triangulation::doesContain(Triangulation::Triangle *tri,
                           const Vec2 &point) const {
  return doesContain(tri->edges[0], point) ||
         doesContain(tri->edges[1], point) ||
         doesContain(tri->edges[2], point) ||
         isInsideTriangle(tri->vert[0]->pos, tri->vert[1]->pos,
                          tri->vert[2]->pos, point);
}

EIGEN_STRONG_INLINE bool Triangulation::isInsideBound(const Vec2 &point) const {
  return isInsideTriangle(_vertices[0]->pos, _vertices[1]->pos,
                          _vertices[2]->pos, point);
}

EIGEN_STRONG_INLINE bool Triangulation::doesContain(Triangulation::Edge *edge,
                                                    const Vec2 &point) const {
  return doesABcontain(edge->vert[0]->pos, edge->vert[1]->pos, point,
                       settingEpsPointIsOnSegment * maxDim);
}

EIGEN_STRONG_INLINE bool Triangulation::isFromBoundingTri(Vertex *vert) const {
  return vert == _vertices[0].get() || vert == _vertices[1].get() ||
         vert == _vertices[2].get();
}

EIGEN_STRONG_INLINE bool Triangulation::isEdgeLegal(Edge *edge) const {
  bool isInc1Bound = isFromBoundingTri(edge->vert[0]);
  bool isInc2Bound = isFromBoundingTri(edge->vert[1]);
  if (isInc1Bound && isInc2Bound)
    return true;

  bool isAp1Bound = isFromBoundingTri(findThirdVert(edge->triang[0], edge));
  bool isAp2Bound = isFromBoundingTri(findThirdVert(edge->triang[1], edge));
  if ((isAp1Bound || isAp2Bound) && !isInc1Bound && !isInc2Bound)
    return true;

  const Vec2 &a = edge->vert[0]->pos;
  const Vec2 &b = edge->vert[1]->pos;
  const Vec2 &c = findThirdVert(edge->triang[0], edge)->pos;
  const Vec2 &d = findThirdVert(edge->triang[1], edge)->pos;

  if (isInc1Bound || isInc2Bound)
    return !isABCDConvex(a, c, b, d);

  return !isABCDConvex(a, c, b, d) || isABLegal(a, b, c, d);
}

Triangulation::Triangle *Triangulation::enclosingTriangle(const Vec2 &point) {
  if (!isInsideBound(point))
    return nullptr;

  // "visibility walk"

  //  std::cout << "look for " << point.transpose() << std::endl;

  Edge *curEdge = *_vertices[0]->edges.begin();
  if (curEdge->triang[0] != nullptr && doesContain(curEdge->triang[0], point))
    return curEdge->triang[0];
  if (curEdge->triang[1] != nullptr && doesContain(curEdge->triang[1], point))
    return curEdge->triang[1];

  Triangle *curTri = nullptr;

  do {
    //    std::cout << "edge from " << curEdge->vert[0]->pos.transpose() << " to
    //    "
    //              << curEdge->vert[1]->pos.transpose() << std::endl;
    int adjTriInd;
    if (curEdge->triang[0] == nullptr)
      adjTriInd = 1;
    else if (curEdge->triang[1] == nullptr)
      adjTriInd = 0;
    else if (isSameSide(curEdge->vert[0]->pos, curEdge->vert[1]->pos,
                        findThirdVert(curEdge->triang[0], curEdge)->pos, point))
      adjTriInd = 0;
    else
      adjTriInd = 1;

    curTri = curEdge->triang[adjTriInd];

    //    std::cout << "point = " << point.transpose() << std::endl;
    //    std::cout << "in on edge? " << doesContain(curEdge, point) <<
    //    std::endl; std::cout << "in on tri? " << doesContain(curTri, point) <<
    //    std::endl;

    Edge *otherEdges[2];
    int otherEdgesInsertInd = 0;
    for (const auto &triSide : curTri->edges) {
      if (curEdge == triSide)
        continue;
      otherEdges[otherEdgesInsertInd++] = triSide;
    }

    bool isPossibleWay[] = {
        !isSameSide(otherEdges[0]->vert[0]->pos, otherEdges[0]->vert[1]->pos,
                    findThirdVert(curTri, otherEdges[0])->pos, point),
        !isSameSide(otherEdges[1]->vert[0]->pos, otherEdges[1]->vert[1]->pos,
                    findThirdVert(curTri, otherEdges[1])->pos, point)};

    //    std::cout << "ways = " << isPossibleWay[0] << " " << isPossibleWay[1]
    //              << std::endl;

    int randomChoice;
    if (isPossibleWay[0] && isPossibleWay[1])
      randomChoice = mt() & 1;
    else
      randomChoice = 1;

    for (int otherEdgesInd = 0; otherEdgesInd < 2; ++otherEdgesInd) {
      if (!isPossibleWay[otherEdgesInd])
        continue;
      if (randomChoice) {
        curEdge = otherEdges[otherEdgesInd];
        break;
      } else
        randomChoice = !randomChoice;
    }
  } while (!doesContain(curTri, point));
  return curTri;
}

bool Triangulation::isIncidentToBoundary(Triangle *tri) const {
  return isFromBoundingTri(tri->vert[0]) || isFromBoundingTri(tri->vert[1]) ||
         isFromBoundingTri(tri->vert[2]);
}

void Triangulation::divideTriangle(Triangle *tri, Vertex *vert) {
  if (!doesContain(tri, vert->pos))
    throw std::runtime_error("triangulation inner problem: tried to insert "
                             "point outside a triangle!");

  Edge *newEdges[] = {makeEdge(), makeEdge(), makeEdge()};
  Triangle *newTri[] = {makeTriangle(), makeTriangle(), makeTriangle()};

  for (int newEdgesInd = 0; newEdgesInd < 3; ++newEdgesInd) {
    const auto &curNewEdge = newEdges[newEdgesInd];
    curNewEdge->vert[0] = vert;
    curNewEdge->vert[1] = tri->vert[newEdgesInd];

    curNewEdge->triang[0] = newTri[(newEdgesInd + 1) % 3];
    curNewEdge->triang[1] = newTri[(newEdgesInd + 2) % 3];
    newTri[(newEdgesInd + 1) % 3]->edges[0] = curNewEdge;
    newTri[(newEdgesInd + 2) % 3]->edges[1] = curNewEdge;
  }

  for (int newTriInd = 0; newTriInd < 3; ++newTriInd) {
    const auto &curNewTri = newTri[newTriInd];
    const auto &externEdge = findOppositeEdge(tri, tri->vert[newTriInd]);
    int edgeCurTriInd = externEdge->triang[0] == tri ? 0 : 1;

    curNewTri->edges[2] = externEdge;
    externEdge->triang[edgeCurTriInd] = curNewTri;

    curNewTri->vert[0] = tri->vert[(newTriInd + 1) % 3];
    curNewTri->vert[1] = tri->vert[(newTriInd + 2) % 3];
    curNewTri->vert[2] = vert;
  }

  for (int triVertInd = 0; triVertInd < 3; ++triVertInd) {
    const auto &curVert = tri->vert[triVertInd];
    curVert->edges.insert(newEdges[triVertInd]);

    curVert->triangles.insert(newTri[(triVertInd + 1) % 3]);
    curVert->triangles.insert(newTri[(triVertInd + 2) % 3]);

    curVert->triangles.erase(tri);
  }

  for (int i = 0; i < 3; ++i) {
    vert->edges.insert(newEdges[i]);
    vert->triangles.insert(newTri[i]);
  }
}

void Triangulation::divideEdge(Edge *edge, Vertex *vert) {
  // indexed as oldTri[apartVertInd][incVertInd]
  Triangle *oldTri[] = {edge->triang[0], edge->triang[1]};
  Vertex *incVert[] = {edge->vert[0], edge->vert[1]};
  Vertex *apartVert[] = {findThirdVert(edge->triang[0], edge),
                         findThirdVert(edge->triang[1], edge)};

  Triangle *newTri[][2] = {{makeTriangle(), makeTriangle()},
                           {makeTriangle(), makeTriangle()}};

  Edge *newIncEdges[] = {makeEdge(), makeEdge()};
  Edge *newApEdges[] = {makeEdge(), makeEdge()};

  for (int apartVertInd = 0; apartVertInd < 2; ++apartVertInd)
    for (int incVertInd = 0; incVertInd < 2; ++incVertInd) {
      const auto &curNewTri = newTri[apartVertInd][incVertInd];
      curNewTri->edges[0] = newApEdges[apartVertInd];
      newApEdges[apartVertInd]->triang[incVertInd] = curNewTri;

      curNewTri->edges[1] = newIncEdges[incVertInd];
      newIncEdges[incVertInd]->triang[apartVertInd] = curNewTri;

      Edge *externEdge = nullptr;
      for (const auto &oldTriSide : oldTri[apartVertInd]->edges) {
        if (oldTriSide == edge)
          continue;
        if (isIncident(incVert[incVertInd], oldTriSide)) {
          externEdge = oldTriSide;
          break;
        }
      }
      curNewTri->edges[2] = externEdge;
      int oldTriEdgeInd = externEdge->triang[0] == oldTri[apartVertInd] ? 0 : 1;
      externEdge->triang[oldTriEdgeInd] = curNewTri;
    }

  for (int apartVertInd = 0; apartVertInd < 2; ++apartVertInd)
    for (int incVertInd = 0; incVertInd < 2; ++incVertInd) {
      const auto &curNewTri = newTri[apartVertInd][incVertInd];
      curNewTri->vert[0] = incVert[incVertInd];
      incVert[incVertInd]->triangles.insert(curNewTri);

      curNewTri->vert[1] = vert;
      vert->triangles.insert(curNewTri);

      curNewTri->vert[2] = apartVert[apartVertInd];
      apartVert[apartVertInd]->triangles.insert(curNewTri);
    }

  for (int apartVertInd = 0; apartVertInd < 2; ++apartVertInd) {
    apartVert[apartVertInd]->triangles.erase(oldTri[apartVertInd]);
    for (int incVertInd = 0; incVertInd < 2; ++incVertInd)
      incVert[incVertInd]->triangles.erase(oldTri[apartVertInd]);
  }

  for (int incEdgeInd = 0; incEdgeInd < 2; ++incEdgeInd) {
    const auto &curIncEdge = newIncEdges[incEdgeInd];
    curIncEdge->vert[0] = vert;
    vert->edges.insert(curIncEdge);

    curIncEdge->vert[1] = incVert[incEdgeInd];
    incVert[incEdgeInd]->edges.insert(newIncEdges[incEdgeInd]);

    incVert[incEdgeInd]->edges.erase(edge);
  }

  for (int apartEdgeInd = 0; apartEdgeInd < 2; ++apartEdgeInd) {
    const auto &curApartEdge = newApEdges[apartEdgeInd];
    curApartEdge->vert[0] = vert;
    vert->edges.insert(curApartEdge);

    curApartEdge->vert[1] = apartVert[apartEdgeInd];
    apartVert[apartEdgeInd]->edges.insert(curApartEdge);
  }
}

Triangulation::Vertex *Triangulation::addPoint(Triangulation::Vertex *newVert) {
  Triangle *tri = enclosingTriangle(newVert->pos);
  std::queue<Edge *> maybeIllegal;

  // std::cout << "add vert " << newVert->pos.transpose() << std::endl;

  for (const auto &vert : tri->vert)
    if (areEqual(vert->pos, newVert->pos, settingEpsSamePoints * maxDim))
      return vert;

  Edge *enclosingSide = nullptr;
  for (const auto &side : tri->edges)
    if (doesContain(side, newVert->pos))
      enclosingSide = side;

  if (enclosingSide) {
    for (const auto &nearTri : enclosingSide->triang)
      for (const auto &nearSide : nearTri->edges) {
        if (nearSide == enclosingSide)
          continue;
        maybeIllegal.push(nearSide);
      }
    divideEdge(enclosingSide, newVert);
  } else {
    for (const auto &side : tri->edges)
      maybeIllegal.push(side);
    divideTriangle(tri, newVert);
  }

  while (!maybeIllegal.empty()) {
    Edge *curEdge = maybeIllegal.front();
    maybeIllegal.pop();

    if (isEdgeLegal(curEdge))
      continue;

    int oppositeTriInd = isIncident(newVert, curEdge->triang[0]) ? 1 : 0;
    Triangle *oppositeTri = curEdge->triang[oppositeTriInd];
    for (const auto &opTriSide : oppositeTri->edges) {
      if (opTriSide == curEdge)
        continue;
      maybeIllegal.push(opTriSide);
    }

    performFlip(curEdge);
  }

  return newVert;
}

void Triangulation::tidy() {
  std::unordered_set<Edge *> edgesUsed;
  std::unordered_set<Triangle *> trianglesUsed;

  for (int i = 0; i < int(_vertices.size()); ++i) {
    Vertex *vert = _vertices[i].get();
    for (Edge *edge : vert->edges)
      edgesUsed.insert(edge);
    for (Triangle *tri : vert->triangles)
      trianglesUsed.insert(tri);
  }

  std::vector<std::unique_ptr<Edge>> newEdges;
  newEdges.reserve(_edges.size());
  std::vector<std::unique_ptr<Triangle>> newTriangles;
  newTriangles.reserve(_triangles.size());
  for (auto &e : _edges) {
    if (edgesUsed.find(e.get()) != edgesUsed.end()) {
      newEdges.push_back(std::unique_ptr<Edge>(nullptr));
      std::swap(newEdges.back(), e);
    }
  }
  for (auto &tri : _triangles) {
    if (trianglesUsed.find(tri.get()) != trianglesUsed.end()) {
      newTriangles.push_back(std::unique_ptr<Triangle>(nullptr));
      std::swap(newTriangles.back(), tri);
    }
  }

  std::swap(_edges, newEdges);
  std::swap(_triangles, newTriangles);
}

void Triangulation::fillPtrVectors() {
  vertexPtrs.reserve(_vertices.size() - 3);
  for (int i = 0; i < int(_vertices.size()) - 3; ++i)
    vertexPtrs.push_back(operator[](i));
  edgePtrs.reserve(_edges.size());
  for (const auto &e : _edges)
    if (!isFromBoundingTri(e->vert[0]) && !isFromBoundingTri(e->vert[1]))
      edgePtrs.push_back(e.get());
  trianglePtrs.reserve(_triangles.size());
  for (const auto &t : _triangles)
    if (!isIncidentToBoundary(t.get()))
      trianglePtrs.push_back(t.get());
}

cv::Mat Triangulation::draw(int imgWidth, int imgHeight) const {
  cv::Mat res(imgHeight, imgWidth, CV_8UC3, CV_BLACK);

  double p = settingTriangulationDrawPadding;

  Vec2 diag = (bottomRight - upperLeft) * (1 + 2 * p);
  double scaleX = double(imgHeight) / diag[0];
  double scaleY = double(imgWidth) / diag[1];

  cv::Point startFrom = cv::Point(int(p * imgWidth), int(p * imgHeight));

  drawScaled(res, scaleX, scaleY, startFrom, CV_GREEN);
  return res;
}

void Triangulation::draw(cv::Mat &img, cv::Scalar edgeCol) const {
  drawScaled(img, 1.0, 1.0, toCvPoint(upperLeft), edgeCol);
}

void Triangulation::drawScaled(cv::Mat &img, double scaleX, double scaleY,
                               cv::Point upperLeftPoint,
                               cv::Scalar edgeCol) const {
  std::set<Edge *> edgesDrawn;
  std::set<Triangle *> trianglesDrawn;

  for (const auto &p : _vertices)
    for (auto e : p->edges) {
      if (isFromBoundingTri(e->vert[0]) || isFromBoundingTri(e->vert[1]))
        continue;
      if (edgesDrawn.find(e) != edgesDrawn.end())
        continue;
      edgesDrawn.insert(e);
      cv::Point v[] = {toCvPoint(e->vert[0]->pos - upperLeft, scaleX, scaleY,
                                 upperLeftPoint),
                       toCvPoint(e->vert[1]->pos - upperLeft, scaleX, scaleY,
                                 upperLeftPoint)};
      cv::line(img, v[0], v[1], edgeCol, 1);
    }

  //  for (auto p : _points) {
  //    cv::Point cvp(int(p->pos[0] * scaleX), int(p->pos[1] * scaleY));
  //    cv::circle(result, cvp, 2, CV_RED, CV_FILLED);
  //  }

  //  for (auto p : _points)
  //    for (auto t : p->triangles) {
  //      if (!t)
  //        continue;
  //      if (trianglesDrawn.find(t) != trianglesDrawn.end())
  //        continue;
  //      trianglesDrawn.insert(t);
  //      Vec2 center = (t->vert[0]->pos + t->vert[1]->pos + t->vert[2]->pos) /
  //      3;

  //      cv::Point cvcenter = toCvPoint(center, scaleX, scaleY);
  //      cv::Point vert[] = {toCvPoint(t->vert[0]->pos, scaleX, scaleY),
  //                          toCvPoint(t->vert[1]->pos, scaleX, scaleY),
  //                          toCvPoint(t->vert[2]->pos, scaleX, scaleY)};
  //      cv::Point edgeCenters[] = {
  //          toCvPoint((t->edges[0]->vert[0]->pos + t->edges[0]->vert[1]->pos)
  //          / 2,
  //                    scaleX, scaleY),
  //          toCvPoint((t->edges[1]->vert[0]->pos + t->edges[1]->vert[1]->pos)
  //          / 2,
  //                    scaleX, scaleY),
  //          toCvPoint((t->edges[2]->vert[0]->pos + t->edges[2]->vert[1]->pos)
  //          / 2,
  //                    scaleX, scaleY)};

  //      for (auto v : vert)
  //        cv::arrowedLine(result, cvcenter, (v + cvcenter) / 2, CV_MAGNETA /
  //        2);
  //      for (auto e : edgeCenters)
  //        cv::arrowedLine(result, cvcenter, (e + cvcenter) / 2, CV_BLUE / 2);
  //    }
}

void drawCurvedInternal(CameraModel *cam, Vec2 ptFrom, Vec2 ptTo, cv::Mat &img,
                        cv::Scalar edgeCol) {
  constexpr int curveSectors = 50;
  constexpr int curvePoints = curveSectors + 1;

  Vec3 rayFrom = cam->unmap(ptFrom.data()).normalized();
  Vec3 rayTo = cam->unmap(ptTo.data()).normalized();
  double fullAngle = std::acos(rayFrom.dot(rayTo));
  Vec3 axis = rayFrom.cross(rayTo).normalized();
  cv::Point pnts[curvePoints];

  for (int it = 0; it <= curveSectors; ++it) {
    double curAngle = fullAngle * it / curveSectors;
    SO3 curRot = SO3::exp(curAngle * axis);
    Vec3 rayCur = curRot * rayFrom;
    pnts[it] = toCvPoint(cam->map(rayCur.data()));
  }

  for (int it = 0; it < curveSectors; ++it)
    cv::line(img, pnts[it], pnts[it + 1], edgeCol, 1);
}

void drawEdgeCurved(CameraModel *cam, Triangulation::Edge *edge, cv::Mat &img,
                    cv::Scalar edgeCol) {
  drawCurvedInternal(cam, edge->vert[0]->pos, edge->vert[1]->pos, img, edgeCol);
}

void Triangulation::drawCurved(CameraModel *cam, cv::Mat &img,
                               cv::Scalar edgeCol) const {
  std::set<Edge *> edgesDrawn;
  for (const auto &p : _vertices)
    for (auto e : p->edges) {
      if (isFromBoundingTri(e->vert[0]) || isFromBoundingTri(e->vert[1]))
        continue;
      if (edgesDrawn.find(e) != edgesDrawn.end())
        continue;
      drawEdgeCurved(cam, e, img, edgeCol);
    }
}

} // namespace fishdso
