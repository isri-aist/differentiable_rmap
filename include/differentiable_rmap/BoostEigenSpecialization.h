// Copied and edited from https://hans-robo.hatenablog.com/entry/2019/09/26/010359

#pragma once

#include <boost/geometry.hpp>
#include <Eigen/Core>


// The following definition allows Eigen::Vector2d to be used as the point type in boost::geometry.
namespace boost::geometry::traits
{
template<>
struct tag<Eigen::Vector2d> {
  typedef point_tag type;
};

template<>
struct coordinate_type<Eigen::Vector2d> {
  typedef double type;
};

template<>
struct coordinate_system<Eigen::Vector2d> {
  typedef cs::cartesian type;
};

template<>
struct dimension<Eigen::Vector2d> : boost::mpl::int_<2> {
};

template<>
struct access<Eigen::Vector2d, 0> {
  static double get(Eigen::Vector2d const &p) {
    return p.x();
  }
  static void set(Eigen::Vector2d &p, double const &value) {
    p.x() = value;
  }
};

template<>
struct access<Eigen::Vector2d, 1> {
  static double get(Eigen::Vector2d const &p) {
    return p.y();
  }

  static void set(Eigen::Vector2d &p, double const &value) {
    p.y() = value;
  }
};
}
