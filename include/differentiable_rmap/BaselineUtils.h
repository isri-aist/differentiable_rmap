/* Author: Masaki Murooka */

#pragma once

#include <vector>
#include <memory>

#include <Eigen/Core>


namespace DiffRmap
{
/** \brief Run one-class nearest neightbor binary classification
    \tparam N sample dimension
    \param test_sample test sample
    \param dist_ratio_thre threshold of distaice ratio
    \param train_sample_list training sample list
    \return true if test_sample is estimated to belong to the positive class

    \note See https://arxiv.org/abs/1604.01686 for algorithm
*/
template <size_t N>
bool oneClassNearestNeighbor(
    const Eigen::Matrix<double, N, 1>& test_sample,
    double dist_ratio_thre,
    const std::vector<Eigen::Matrix<double, N, 1>>& train_sample_list);

/** \brief Run k-nearest neightbor binary classification
    \tparam N sample dimension
    \param test_sample test sample
    \param K number of nearest points
    \param train_sample_list training sample list
    \param class_list training class list (true/false for positive/negative class)
    \return true if test_sample is estimated to belong to the positive class
*/
template <size_t N>
bool kNearestNeighbor(
    const Eigen::Matrix<double, N, 1>& test_sample,
    size_t K,
    const std::vector<Eigen::Matrix<double, N, 1>>& train_sample_list,
    const std::vector<bool>& class_list);


/** \brief Class that classifies whether a point is inside or outside a convex. */
class ConvexInsideClassification
{
 protected:
  /** \brief Implementation class to hide boost codes. */
  class Impl;

 public:
  /** \brief Constructor.
      \param points training points to make convex
  */
  ConvexInsideClassification(const std::vector<Eigen::Vector2d>& points);

  /** \brief Destructor. */
  ~ConvexInsideClassification();

  /** \brief Classify point.
      \param test point
  */
  bool classify(const Eigen::Vector2d& point) const;

 protected:
  //! Implementation
  std::unique_ptr<Impl> impl_;
};
}

// See method 3 in https://www.codeproject.com/Articles/48575/How-to-Define-a-Template-Class-in-a-h-File-and-Imp
#include <differentiable_rmap/BaselineUtils.hpp>
