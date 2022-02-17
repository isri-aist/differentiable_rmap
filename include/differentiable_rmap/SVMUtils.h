/* Author: Masaki Murooka */

#pragma once

#include <libsvm/svm.h>

#include <differentiable_rmap/SamplingUtils.h>

namespace DiffRmap
{
/** \brief Set SVM matrix for prediction.
    \tparam SamplingSpaceType sampling space
    \param[out] svm_coeff_vec support vector coefficients
    \param[out] svm_sv_mat support vector matrix
    \param[in] svm_mo SVM model
*/
template<SamplingSpace SamplingSpaceType>
void setSVMPredictionMat(Eigen::Ref<Eigen::VectorXd> svm_coeff_vec,
                         Eigen::Ref<Eigen::Matrix<double, inputDim<SamplingSpaceType>(), Eigen::Dynamic>> svm_sv_mat,
                         svm_model * svm_mo);

/** \brief Set SVM input to node (including index).
    \tparam SamplingSpaceType sampling space
    \param[out] input_node SVM input node
    \param[in] input SVM input
*/
template<SamplingSpace SamplingSpaceType>
void setInputNode(svm_node * input_node, const Input<SamplingSpaceType> & input);

/** \brief Set SVM input to node (only value).
    \tparam SamplingSpaceType sampling space
    \param[out] input_node SVM input node
    \param[in] input SVM input
*/
template<SamplingSpace SamplingSpaceType>
void setInputNodeOnlyValue(svm_node * input_node, const Input<SamplingSpaceType> & input);

/** \brief Convert SVM input node to Eigen::Vector.
    \tparam SamplingSpaceType sampling space
    \param input_node SVM input node
*/
template<SamplingSpace SamplingSpaceType>
Input<SamplingSpaceType> svmNodeToEigenVec(const svm_node * input_node);

/** \brief Calculate SVM value.
    \tparam SamplingSpaceType sampling space
    \param sample sample
    \param svm_param SVM parameter
    \param svm_mo SVM model
    \param svm_coeff_vec support vector coefficients
    \param svm_sv_mat support vector matrix
    \return predicted SVM value
*/
template<SamplingSpace SamplingSpaceType>
double calcSVMValue(const Sample<SamplingSpaceType> & sample,
                    const svm_parameter & svm_param,
                    svm_model * svm_mo,
                    const Eigen::VectorXd & svm_coeff_vec,
                    const Eigen::Matrix<double, inputDim<SamplingSpaceType>(), Eigen::Dynamic> & svm_sv_mat);

/** \brief Calculate gradient of SVM value.
    \tparam SamplingSpaceType sampling space
    \param sample sample
    \param svm_param SVM parameter
    \param svm_mo SVM model
    \param svm_coeff_vec support vector coefficients
    \param svm_sv_mat support vector matrix
    \return gradient of predicted SVM value (column vector)
*/
template<SamplingSpace SamplingSpaceType>
Sample<SamplingSpaceType> calcSVMGrad(
    const Sample<SamplingSpaceType> & sample,
    const svm_parameter & svm_param,
    svm_model * svm_mo,
    const Eigen::VectorXd & svm_coeff_vec,
    const Eigen::Matrix<double, inputDim<SamplingSpaceType>(), Eigen::Dynamic> & svm_sv_mat);

/*! \brief Type of matrix to represent the linear relation from input to sample. */
template<SamplingSpace SamplingSpaceType>
using InputToSampleMat = Eigen::Matrix<double, sampleDim<SamplingSpaceType>(), inputDim<SamplingSpaceType>()>;

/*! \brief Type of matrix to represent the linear relation from sample to vel. */
template<SamplingSpace SamplingSpaceType>
using SampleToVelMat = Eigen::Matrix<double, velDim<SamplingSpaceType>(), sampleDim<SamplingSpaceType>()>;

/** \brief Get matrix to convert gradient for input to gradient for sample. Gradient is assumed to be column vector.
    \tparam SamplingSpaceType sampling space
    \param sample sample
*/
template<SamplingSpace SamplingSpaceType>
InputToSampleMat<SamplingSpaceType> inputToSampleMat(const Sample<SamplingSpaceType> & sample);

/** \brief Get matrix to convert gradient for sample to gradient for vel. Gradient is assumed to be column vector.
    \tparam SamplingSpaceType sampling space
    \param sample sample
*/
template<SamplingSpace SamplingSpaceType>
SampleToVelMat<SamplingSpaceType> sampleToVelMat(const Sample<SamplingSpaceType> & sample);

/** \brief Get relative sample between two samples which is represented in predecessor frame.
    \tparam SamplingSpaceType sampling space
    \param pre_sample predecessor sample
    \param suc_sample successor sample
*/
template<SamplingSpace SamplingSpaceType>
Sample<SamplingSpaceType> relSample(const Sample<SamplingSpaceType> & pre_sample,
                                    const Sample<SamplingSpaceType> & suc_sample);

/** \brief Get middle sample between two samples.
    \tparam SamplingSpaceType sampling space
    \param sample1 first sample
    \param sample2 second sample
*/
template<SamplingSpace SamplingSpaceType>
Sample<SamplingSpaceType> midSample(const Sample<SamplingSpaceType> & sample1,
                                    const Sample<SamplingSpaceType> & sample2);

/*! \brief Type of matrix to represent the linear relation from sample to sample. */
template<SamplingSpace SamplingSpaceType>
using SampleToSampleMat = Eigen::Matrix<double, sampleDim<SamplingSpaceType>(), sampleDim<SamplingSpaceType>()>;

/** \brief Get matrix to represent the linear relation from relative velocity to single velocity.
    \tparam SamplingSpaceType sampling space
    \param pre_sample predecessor sample
    \param suc_sample successor sample
    \param wrt_suc if true, the returned matrix is w.r.t. the successor sample. otherwise, it is w.r.t. predecessor
   sample.
*/
template<SamplingSpace SamplingSpaceType>
SampleToSampleMat<SamplingSpaceType> relSampleToSampleMat(const Sample<SamplingSpaceType> & pre_sample,
                                                          const Sample<SamplingSpaceType> & suc_sample,
                                                          bool wrt_suc);
} // namespace DiffRmap

// See method 3 in https://www.codeproject.com/Articles/48575/How-to-Define-a-Template-Class-in-a-h-File-and-Imp
#include <differentiable_rmap/SVMUtils.hpp>
