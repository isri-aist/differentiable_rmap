/* Author: Masaki Murooka */

#include <boost/geometry/geometries/multi_point.hpp>
#include <boost/geometry/geometries/polygon.hpp>

#include <differentiable_rmap/BaselineUtils.h>
#include <differentiable_rmap/BoostEigenSpecialization.h>

using namespace DiffRmap;


class ConvexInsideClassification::Impl
{
 protected:
  /** \brief Multi-point class specialized with Eigen vector. */
  using BGMultiPointType = boost::geometry::model::multi_point<Eigen::Vector2d>;

  /** \brief Polygon class specialized with Eigen vector. */
  using BGPolygonType = boost::geometry::model::polygon<Eigen::Vector2d>;

 public:
  Impl(const std::vector<Eigen::Vector2d>& points)
  {
    for (const auto& point : points) {
      boost::geometry::append(bg_points_, point);
    }

    boost::geometry::convex_hull(bg_points_, bg_hull_);
  }

  bool classify(const Eigen::Vector2d& point) const
  {
    return boost::geometry::within(point, bg_hull_);
  }

  void setConvexVertices(std::vector<Eigen::Vector2d>& convex_vertices) const
  {
    convex_vertices.clear();
    boost::geometry::for_each_point(
        bg_hull_.outer(),
        [&convex_vertices](const Eigen::Vector2d& point) {
          convex_vertices.push_back(point);
        });
  }

 protected:
  //! Training points
  BGMultiPointType bg_points_;

  //! Convex hull of training points
  BGPolygonType bg_hull_;
};

ConvexInsideClassification::ConvexInsideClassification(
    const std::vector<Eigen::Vector2d>& points)
{
  impl_.reset(new Impl(points));
  impl_->setConvexVertices(convex_vertices_);
}

ConvexInsideClassification::~ConvexInsideClassification()
{
  impl_.reset(nullptr);
}

bool ConvexInsideClassification::classify(const Eigen::Vector2d& point) const
{
  return impl_->classify(point);
}
