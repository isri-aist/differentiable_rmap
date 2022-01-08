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
    const Input<SamplingSpaceType>& input);

/** \brief Set SVM input to node (only value).
    \tparam SamplingSpaceType sampling space
    \param[out] input_node SVM input node
    \param[in] input SVM input
*/
template <SamplingSpace SamplingSpaceType>
void setInputNodeOnlyValue(
    svm_node* input_node,
    const Input<SamplingSpaceType>& input);

/** \brief Convert SVM input node to Eigen::Vector.
    \tparam SamplingSpaceType sampling space
    \param input_node SVM input node
*/
template <SamplingSpace SamplingSpaceType>
Input<SamplingSpaceType> svmNodeToEigenVec(
    const svm_node *input_node);

/** \brief Calculate SVM value.
    \tparam SamplingSpaceType sampling space
    \param sample sample
    \param svm_param SVM parameter
    \param svm_mo SVM model
    \param svm_coeff_vec support vector coefficients
    \param svm_sv_mat support vector matrix
    \return predicted SVM value
*/
template <SamplingSpace SamplingSpaceType>
double calcSVMValue(
    const Sample<SamplingSpaceType>& sample,
    const svm_parameter& svm_param,
    svm_model *svm_mo,
    const Eigen::VectorXd& svm_coeff_vec,
    const Eigen::Matrix<double, inputDim<SamplingSpaceType>(), Eigen::Dynamic>& svm_sv_mat);

/** \brief Calculate gradient of SVM value.
    \tparam SamplingSpaceType sampling space
    \param sample sample
    \param svm_param SVM parameter
    \param svm_mo SVM model
    \param svm_coeff_vec support vector coefficients
    \param svm_sv_mat support vector matrix
    \return gradient of predicted SVM value (column vector)
*/
template <SamplingSpace SamplingSpaceType>
Vel<SamplingSpaceType> calcSVMGrad(
    const Sample<SamplingSpaceType>& sample,
    const svm_parameter& svm_param,
    svm_model *svm_mo,
    const Eigen::VectorXd& svm_coeff_vec,
    const Eigen::Matrix<double, inputDim<SamplingSpaceType>(), Eigen::Dynamic>& svm_sv_mat);

/** \brief Get matrix to convert gradient for input to gradient for vel. Gradient is assumed to be column vector.
    \tparam SamplingSpaceType sampling space
    \param sample sample
*/
template <SamplingSpace SamplingSpaceType>
Eigen::Matrix<double, velDim<SamplingSpaceType>(), inputDim<SamplingSpaceType>()>
inputToVelMat(const Sample<SamplingSpaceType>& sample);

// /** \brief Set inequality matrix and vecor for SVM constraint in QP.
//     \tparam SamplingSpaceType sampling space
//     \param ineq_mat[out] inequality matrix
//     \param ineq_vec[out] inequality vector
// */
// template <SamplingSpace SamplingSpaceType>
// void setSVMIneq(Eigen::Ref<Eigen::MatrixXd> ineq_mat,
//                 Eigen::Ref<Eigen::MatrixXd> ineq_vec,
//                 const Sample<SamplingSpaceType>& sample,
//                 const svm_parameter& svm_param,
//                 svm_model *svm_mo,
//                 const Eigen::VectorXd& svm_coeff_vec,
//                 const Eigen::Matrix<double, inputDim<SamplingSpaceType>(), Eigen::Dynamic>& svm_sv_mat,
//                 double svm_thre)
// {
//   // There is a problem with receiving a fixed size matrix with Ref, so we receive a dynamic size matrix.
//   // See https://stackoverflow.com/a/54966664
//   assert(ineq_mat.rows() == 1);
//   assert(ineq_mat.cols() == sampleDim<SamplingSpaceType>());
//   assert(ineq_vec.rows() == 1);
//   assert(ineq_vec.cols() == 1);

//   Eigen::VectorXd svm_grad = calcSVMGrad<SamplingSpaceType>(sample, svm_param, svm_mo, svm_coeff_vec, svm_sv_mat);
//   // \todo: transform rotation to theta
//   ineq_mat = -1 * svm_grad.transpose();
//   ineq_vec(0, 0) = calcSVMValue<SamplingSpaceType>(sample, svm_param, svm_mo, svm_coeff_vec, svm_sv_mat) - svm_thre;
// }
}

// See method 3 in https://www.codeproject.com/Articles/48575/How-to-Define-a-Template-Class-in-a-h-File-and-Imp
#include <differentiable_rmap/SVMUtils.hpp>
