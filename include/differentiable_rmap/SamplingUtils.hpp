/* Author: Masaki Murooka */

#include <differentiable_rmap/MathUtils.h>


namespace DiffRmap
{
template <SamplingSpace SamplingSpaceType>
constexpr int sampleDim()
{
  mc_rtc::log::error_and_throw<std::runtime_error>(
      "[sampleDim] Need to be specialized for {}.", std::to_string(SamplingSpaceType));
}

template <>
inline constexpr int sampleDim<SamplingSpace::R2>()
{
  return 2;
}

template <>
inline constexpr int sampleDim<SamplingSpace::SO2>()
{
  return 1;
}

template <>
inline constexpr int sampleDim<SamplingSpace::SE2>()
{
  return sampleDim<SamplingSpace::R2>() + sampleDim<SamplingSpace::SO2>();
}

template <>
inline constexpr int sampleDim<SamplingSpace::R3>()
{
  return 3;
}

template <>
inline constexpr int sampleDim<SamplingSpace::SO3>()
{
  return 4;
}

template <>
inline constexpr int sampleDim<SamplingSpace::SE3>()
{
  return sampleDim<SamplingSpace::R3>() + sampleDim<SamplingSpace::SO3>();
}

template <SamplingSpace SamplingSpaceType>
Eigen::Matrix<double, sampleDim<SamplingSpaceType>(), 1> poseToSample(
    const sva::PTransformd& pose)
{
  mc_rtc::log::error_and_throw<std::runtime_error>(
      "[poseToSample] Need to be specialized for {}.", std::to_string(SamplingSpaceType));
}

template <>
inline Eigen::Matrix<double, sampleDim<SamplingSpace::R2>(), 1> poseToSample<SamplingSpace::R2>(
    const sva::PTransformd& pose)
{
  return pose.translation().head(2);
}

template <>
inline Eigen::Matrix<double, sampleDim<SamplingSpace::SO2>(), 1> poseToSample<SamplingSpace::SO2>(
    const sva::PTransformd& pose)
{
  Eigen::Matrix<double, sampleDim<SamplingSpace::SO2>(), 1> sample;
  sample << calcYawAngle(pose.rotation().transpose());
  return sample;
}

template <>
inline Eigen::Matrix<double, sampleDim<SamplingSpace::SE2>(), 1> poseToSample<SamplingSpace::SE2>(
    const sva::PTransformd& pose)
{
  Eigen::Matrix<double, sampleDim<SamplingSpace::SE2>(), 1> sample;
  sample << poseToSample<SamplingSpace::R2>(pose), poseToSample<SamplingSpace::SO2>(pose);
  return sample;
}

template <>
inline Eigen::Matrix<double, sampleDim<SamplingSpace::R3>(), 1> poseToSample<SamplingSpace::R3>(
    const sva::PTransformd& pose)
{
  return pose.translation();
}

template <>
inline Eigen::Matrix<double, sampleDim<SamplingSpace::SO3>(), 1> poseToSample<SamplingSpace::SO3>(
    const sva::PTransformd& pose)
{
  // Element order is (x, y, z, w)
  Eigen::Matrix<double, sampleDim<SamplingSpace::SO3>(), 1> sample;
  sample << Eigen::Quaterniond(pose.rotation().transpose()).coeffs();
  return sample;
}

template <>
inline Eigen::Matrix<double, sampleDim<SamplingSpace::SE3>(), 1> poseToSample<SamplingSpace::SE3>(
    const sva::PTransformd& pose)
{
  Eigen::Matrix<double, sampleDim<SamplingSpace::SE3>(), 1> sample;
  sample << poseToSample<SamplingSpace::R3>(pose), poseToSample<SamplingSpace::SO3>(pose);
  return sample;
}
}
