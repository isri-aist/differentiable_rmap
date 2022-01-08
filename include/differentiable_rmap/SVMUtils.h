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

/*! \brief Type of matrix to represent the linear relation from input to vel. */
template <SamplingSpace SamplingSpaceType>
using InputToVelMat = Eigen::Matrix<double, velDim<SamplingSpaceType>(), inputDim<SamplingSpaceType>()>;

/** \brief Get matrix to convert gradient for input to gradient for vel. Gradient is assumed to be column vector.
    \tparam SamplingSpaceType sampling space
    \param sample sample
*/
template <SamplingSpace SamplingSpaceType>
InputToVelMat<SamplingSpaceType> inputToVelMat(const Sample<SamplingSpaceType>& sample);

/** \brief Set inequality matrix and vecor for SVM constraint in QP.
    \tparam SamplingSpaceType sampling space
    \param[out] ineq_mat inequality matrix
    \param[out] ineq_vec inequality vector
    \param[in] sample sample
    \param[in] svm_param SVM parameter
    \param[in] svm_mo SVM model
    \param[in] svm_coeff_vec support vector coefficients
    \param[in] svm_sv_mat support vector matrix
    \param[in] svm_thre threshold of SVM predict value to be determined as reachable
*/
template <SamplingSpace SamplingSpaceType>
void setSVMIneq(Eigen::Ref<Eigen::MatrixXd> ineq_mat,
                Eigen::Ref<Eigen::MatrixXd> ineq_vec,
                const Sample<SamplingSpaceType>& sample,
                const svm_parameter& svm_param,
                svm_model *svm_mo,
                const Eigen::VectorXd& svm_coeff_vec,
                const Eigen::Matrix<double, inputDim<SamplingSpaceType>(), Eigen::Dynamic>& svm_sv_mat,
                double svm_thre);
}

// See method 3 in https://www.codeproject.com/Articles/48575/How-to-Define-a-Template-Class-in-a-h-File-and-Imp
#include <differentiable_rmap/SVMUtils.hpp>
