/* Author: Masaki Murooka */

#pragma once


namespace DiffRmap
{
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
}

// See method 3 in https://www.codeproject.com/Articles/48575/How-to-Define-a-Template-Class-in-a-h-File-and-Imp
#include <differentiable_rmap/BaselineUtils.hpp>
