/* Author: Masaki Murooka */

#pragma once

#include <string>
#include <stdexcept>

#include <SpaceVecAlg/SpaceVecAlg>
#include <mc_rtc/logging.h>


namespace DiffRmap
{
/** \brief Sampling space. */
enum class SamplingSpace
{
  R2 = 21,
  SO2 = 22,
  SE2 = 23,
  R3 = 31,
  SO3 = 32,
  SE3 = 33
};

/** \brief Convert string to sampling space. */
SamplingSpace strToSamplingSpace(const std::string& sampling_space_str);

/** \brief Get dimension of sample.
    \tparam SamplingSpaceType sampling space
*/
template <SamplingSpace SamplingSpaceType>
constexpr int sampleDim();

/** \brief Convert pose to sample.
    \tparam SamplingSpaceType sampling space
    \param[in] pose pose
    \return sample (fixed size Eigen::Vector)
 */
template <SamplingSpace SamplingSpaceType>
Eigen::Matrix<double, sampleDim<SamplingSpaceType>(), 1> poseToSample(
    const sva::PTransformd& pose);

/** \brief Convert sample to pointcloud position.
    \tparam SamplingSpaceType sampling space
    \param[in] sample sample
    \return pointcloud position (Eigen::Vector3d)
 */
template <SamplingSpace SamplingSpaceType>
Eigen::Vector3d sampleToCloudPos(
    const Eigen::Matrix<double, sampleDim<SamplingSpaceType>(), 1>& sample);
}

namespace std
{
using DiffRmap::SamplingSpace;

inline string to_string(SamplingSpace sampling_space)
{
  if (sampling_space == SamplingSpace::R2) {
    return std::string("R2");
  } else if (sampling_space == SamplingSpace::SO2) {
    return std::string("SO2");
  } else if (sampling_space == SamplingSpace::SE2) {
    return std::string("SE2");
  } else if (sampling_space == SamplingSpace::R3) {
    return std::string("R3");
  } else if (sampling_space == SamplingSpace::SO3) {
    return std::string("SO3");
  } else if (sampling_space == SamplingSpace::SE3) {
    return std::string("SE3");
  } else {
    mc_rtc::log::error_and_throw<std::runtime_error>(
        "[to_string] Unsupported SamplingSpace: {}", static_cast<int>(sampling_space));
  }
}
}

// See method 3 in https://www.codeproject.com/Articles/48575/How-to-Define-a-Template-Class-in-a-h-File-and-Imp
#include <differentiable_rmap/SamplingUtils.hpp>
