/* Author: Masaki Murooka */

#include <mc_rtc/logging.h>

#include <differentiable_rmap/SamplingUtils.h>

using namespace DiffRmap;

SamplingSpace DiffRmap::strToSamplingSpace(const std::string & sampling_space_str)
{
  if(sampling_space_str == "R2")
  {
    return SamplingSpace::R2;
  }
  else if(sampling_space_str == "SO2")
  {
    return SamplingSpace::SO2;
  }
  else if(sampling_space_str == "SE2")
  {
    return SamplingSpace::SE2;
  }
  else if(sampling_space_str == "R3")
  {
    return SamplingSpace::R3;
  }
  else if(sampling_space_str == "SO3")
  {
    return SamplingSpace::SO3;
  }
  else if(sampling_space_str == "SE3")
  {
    return SamplingSpace::SE3;
  }
  else
  {
    mc_rtc::log::error_and_throw<std::runtime_error>("[strToSamplingSpace] Unsupported SamplingSpace name: {}",
                                                     sampling_space_str);
  }
}
