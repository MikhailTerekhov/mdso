#include "util/defs.h"
#include "util/Triangulation.h"
#include "util/geometry.h"
#include <gtest/gtest.h>
#include <set>

using namespace fishdso;

class TriangulationTest
    : public ::testing::TestWithParam<std::shared_ptr<Triangulation>> {};

TEST_P(TriangulationTest, IsConsistent) {
  const Triangulation &tester = *GetParam();

  std::set<Triangulation::Edge *> edges;
  std::set<Triangulation::Triangle *> triangles;

  for (auto p : tester) {
    for (auto e : p->edges)
      edges.insert(e);
    for (auto t : p->triangles)
      triangles.insert(t);
  }

  for (auto e : edges)
    for (auto p : e->vert)
      EXPECT_NE(p->edges.find(e), p->edges.end());

  for (auto t : triangles)
    for (auto p : t->vert)
      EXPECT_NE(p->triangles.find(t), p->triangles.end());

  for (auto t : triangles)
    for (auto e : t->edges)
      EXPECT_NE(std::find(e->triang, e->triang + 2, t), (e->triang + 2));
}

TEST_P(TriangulationTest, IsPlanar) {
  const Triangulation &tester = *GetParam();

  for (auto e1 : tester.edges())
    for (auto e2 : tester.edges()) {
      if (e1 == e2)
        continue;
      bool test = doesABIntersectCD(e1->vert[0]->pos, e1->vert[1]->pos,
                                     e2->vert[0]->pos, e2->vert[1]->pos);
      if (test) {
        cv::Mat img = tester.draw(800, 800, CV_WHITE, CV_BLACK);
        cv::imshow("failed tri", img);
        cv::waitKey();
      }

      ASSERT_FALSE(test)
          << "these do intersect:\n"
          << "a = " << e1->vert[0]->pos.transpose()
          << " b = " << e1->vert[1]->pos.transpose() << "\n"
          << "and\n"
          << "a = " << e2->vert[0]->pos.transpose()
          << " b = " << e2->vert[1]->pos.transpose() << "\n";
    }
}

std::shared_ptr<Triangulation> getSimpleTriang() {
  StdVector<Vec2> pnt;

  pnt.push_back(Vec2(1, 1));
  pnt.push_back(Vec2(1, 2));
  pnt.push_back(Vec2(2, 1));
  pnt.push_back(Vec2(2, 2));
  pnt.push_back(Vec2(1.5, 1));
  pnt.push_back(Vec2(1.5, 1.5));
  pnt.push_back(Vec2(1.75, 1.75));
  pnt.push_back(Vec2(1.75, 1.5));

  std::shared_ptr<Triangulation> triang(new Triangulation(pnt));
  std::cout << "points:" << std::endl;
  for (auto p : *triang) {
    std::cout << p->pos.transpose() << std::endl;
  }

  std::cout << "edges:" << std::endl;
  for (auto e : triang->edges())
    std::cout << e->vert[0]->pos.transpose() << " to "
              << e->vert[1]->pos.transpose() << std::endl;

  // cv::Mat img = triang->draw(800, 800);
  // cv::imshow("simple", img);
  // cv::waitKey();
  return triang;
}

std::shared_ptr<Triangulation> getRandomTriang() {
  const int pntCount = 1000;
  StdVector<Vec2> pnt;
  pnt.reserve(pntCount);

  std::mt19937 mt;
  std::uniform_real_distribution<double> d(0, 10);

  for (int i = 0; i < pntCount; ++i)
    pnt.push_back(Vec2(d(mt), d(mt)));

  return std::shared_ptr<Triangulation>(new Triangulation(pnt));
}

std::shared_ptr<Triangulation> getNonGeneralTriang() {
  const int segmentsCount = 200;
  const int onSegmCount = 5;
  StdVector<Vec2> pnt;

  std::mt19937 mt;
  std::uniform_real_distribution<double> d(0, 100);
  std::uniform_real_distribution<double> d01(0, 1);

  for (int i = 0; i < segmentsCount; ++i) {
    Vec2 a(d(mt), d(mt)), b(d(mt), d(mt));
    for (int j = 0; j < onSegmCount; ++j) {
      double alpha = d01(mt);
      pnt.push_back(alpha * a + (1 - alpha) * b);
    }
  }

  return std::shared_ptr<Triangulation>(new Triangulation(pnt));
}

INSTANTIATE_TEST_CASE_P(Instantiation, TriangulationTest,
                        ::testing::Values(getSimpleTriang(), getRandomTriang(),
                                          getNonGeneralTriang()));

TEST(TriangulationTest, IndicesConsistent) {
  const int pntCount = 200;
  std::mt19937 mt;
  std::uniform_real_distribution<double> d(0, 10);
  StdVector<Vec2> points;

  for (int i = 0; i < pntCount; ++i)
    points.push_back(Vec2(d(mt), d(mt)));

  Triangulation tester(points);

  for (int i = 0; i < int(points.size()); ++i) {
    auto p = tester[i];
    EXPECT_TRUE(points[i].isApprox(p->pos));
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  //::testing::GTEST_FLAG(filter) = "TriangulationTest.IndicesConsistent";
  return RUN_ALL_TESTS();
}
