/* Author: Masaki Murooka */

#pragma once

#include <differentiable_rmap/SamplingUtils.h>


namespace DiffRmap
{
/** \brief Set sample to SVM input node (including index).
    \tparam SamplingSpaceType sampling space
    \param[out] input_node SVM input node
    \param[in] sample sample
*/
template <SamplingSpace SamplingSpaceType>
void setInputNode(
    svm_node* input_node,
    const Eigen::Matrix<double, inputDim<SamplingSpaceType>(), 1>& sample)
{
  for (int i = 0; i < inputDim<SamplingSpaceType>() + 1; i++) {
    if (i == inputDim<SamplingSpaceType>()) {
      input_node[i].index = -1; // last index must be -1
    } else {
      input_node[i].index = i + 1; // index starts from 1
      input_node[i].value = sample[i];
    }
  }
}

/** \brief Set sample to SVM input node (only value).
    \tparam SamplingSpaceType sampling space
    \param[out] input_node SVM input node
    \param[in] sample sample
*/
template <SamplingSpace SamplingSpaceType>
void setInputNodeOnlyValue(
    svm_node* input_node,
    const Eigen::Matrix<double, inputDim<SamplingSpaceType>(), 1>& sample)
{
  for (int i = 0; i < inputDim<SamplingSpaceType>(); i++) {
    input_node[i].value = sample[i];
  }
}
}
