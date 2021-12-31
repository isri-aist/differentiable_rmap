/* Author: Masaki Murooka */

#pragma once

#include <string>
#include <stdexcept>


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

/** \brief Return whether sampling space is on 2D or not. */
inline bool is2DSamplingSpace(SamplingSpace sampling_space)
{
  return static_cast<int>(sampling_space) / 10 == 2;
}

/** \brief Convert string to sampling space. */
SamplingSpace strToSamplingSpace(const std::string& sampling_space_str);

/** \brief Convert pose to sample.
    \param pose input pose
    \param sampling_space sampling space
    \return sample (Eigen::VectorXd)
 */
Eigen::VectorXd poseToSample(const sva::PTransformd& pose,
                             SamplingSpace sampling_space);
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
    throw std::runtime_error("Unsupported SamplingSpace: " +
                             std::to_string(static_cast<int>(sampling_space)));
  }
}
}
