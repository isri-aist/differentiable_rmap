/* Author: Masaki Murooka */

#pragma once

#include <libsvm/svm.h>

#include <differentiable_rmap/SamplingUtils.h>


namespace DiffRmap
{
/** \brief Set SVM input to node (including index).
    \tparam SamplingSpaceType sampling space
    \param[out] input_node SVM input node
    \param[in] input SVM input
*/
template <SamplingSpace SamplingSpaceType>
void setInputNode(
    svm_node* input_node,
    const Input<SamplingSpaceType>& input)
{
  for (int i = 0; i < inputDim<SamplingSpaceType>() + 1; i++) {
    if (i == inputDim<SamplingSpaceType>()) {
      input_node[i].index = -1; // last index must be -1
    } else {
      input_node[i].index = i + 1; // index starts from 1
      input_node[i].value = input[i];
    }
  }
}

/** \brief Set SVM input to node (only value).
    \tparam SamplingSpaceType sampling space
    \param[out] input_node SVM input node
    \param[in] input SVM input
*/
template <SamplingSpace SamplingSpaceType>
void setInputNodeOnlyValue(
    svm_node* input_node,
    const Input<SamplingSpaceType>& input)
{
  for (int i = 0; i < inputDim<SamplingSpaceType>(); i++) {
    input_node[i].value = input[i];
  }
}

/** \brief Convert SVM input node to Eigen::Vector.
    \tparam SamplingSpaceType sampling space
    \param input_node SVM input node
*/
template <SamplingSpace SamplingSpaceType>
Input<SamplingSpaceType> toEigenVector(
    const svm_node *input_node)
{
  Input<SamplingSpaceType> input;
  for (int i = 0; i < inputDim<SamplingSpaceType>(); i++) {
    input[i] = input_node[i].value;
  }
  return input;
}

/** \brief Calculate SVM value.
    \tparam SamplingSpaceType sampling space
    \param input SVM input
    \param svm_param SVM parameter
    \param svm_mo SVM model
    \param svm_coeff_vec support vector coefficients
    \param svm_sv_mat support vector matrix
    \return predicted SVM value
*/
template <SamplingSpace SamplingSpaceType>
double calcSVMValue(
    const Input<SamplingSpaceType>& input,
    const svm_parameter& svm_param,
    svm_model *svm_mo,
    const Eigen::VectorXd& svm_coeff_vec,
    const Eigen::Matrix<double, inputDim<SamplingSpaceType>(), Eigen::Dynamic>& svm_sv_mat)
{
  if (!(svm_mo->param.svm_type == ONE_CLASS || svm_mo->param.svm_type == NU_SVC)) {
    mc_rtc::log::error_and_throw<std::runtime_error>(
        "[calcSVMValue] Only one-class or nu-svc SVM is supported: {}", svm_mo->param.svm_type);
  }

  if (svm_param.kernel_type != RBF) {
    mc_rtc::log::error_and_throw<std::runtime_error>(
        "[calcSVMValue] Only RBF kernel is supported: {}", svm_param.kernel_type);
  }

  return svm_coeff_vec.dot(
      (-svm_param.gamma * (svm_sv_mat.colwise() - input).colwise().squaredNorm()).array().exp().matrix()) - svm_mo->rho[0];
}

/** \brief Calculate gradient of SVM value.
    \tparam SamplingSpaceType sampling space
    \param input SVM input
    \param svm_param SVM parameter
    \param svm_mo SVM model
    \param svm_coeff_vec support vector coefficients
    \param svm_sv_mat support vector matrix
    \return gradient of predicted SVM value (column vector)
*/
template <SamplingSpace SamplingSpaceType>
Input<SamplingSpaceType> calcSVMGrad(
    const Input<SamplingSpaceType>& input,
    const svm_parameter& svm_param,
    svm_model *svm_mo,
    const Eigen::VectorXd& svm_coeff_vec,
    const Eigen::Matrix<double, inputDim<SamplingSpaceType>(), Eigen::Dynamic>& svm_sv_mat)
{
  if (!(svm_mo->param.svm_type == ONE_CLASS || svm_mo->param.svm_type == NU_SVC)) {
    mc_rtc::log::error_and_throw<std::runtime_error>(
        "[calcSVMGrad] Only one-class or nu-svc SVM is supported: {}", svm_mo->param.svm_type);
  }

  if (svm_param.kernel_type != RBF) {
    mc_rtc::log::error_and_throw<std::runtime_error>(
        "[calcSVMGrad] Only RBF kernel is supported: {}", svm_param.kernel_type);
  }

  Eigen::Matrix<double, inputDim<SamplingSpaceType>(), Eigen::Dynamic> sv_mat_minus_input =
      svm_sv_mat.colwise() - input;

  return 2 * svm_param.gamma * sv_mat_minus_input *
      svm_coeff_vec.cwiseProduct((-svm_param.gamma * sv_mat_minus_input.colwise().squaredNorm()).array().exp().matrix().transpose());
}
}
