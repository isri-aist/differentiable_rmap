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
constexpr int inputDim()
{
  mc_rtc::log::error_and_throw<std::runtime_error>(
      "[inputDim] Need to be specialized for {}.", std::to_string(SamplingSpaceType));
}

template <>
inline constexpr int inputDim<SamplingSpace::R2>()
{
  return 2;
}

template <>
inline constexpr int inputDim<SamplingSpace::SO2>()
{
  return 4;
}

template <>
inline constexpr int inputDim<SamplingSpace::SE2>()
{
  return inputDim<SamplingSpace::R2>() + inputDim<SamplingSpace::SO2>();
}

template <>
inline constexpr int inputDim<SamplingSpace::R3>()
{
  return 3;
}

template <>
inline constexpr int inputDim<SamplingSpace::SO3>()
{
  return 9;
}

template <>
inline constexpr int inputDim<SamplingSpace::SE3>()
{
  return inputDim<SamplingSpace::R3>() + inputDim<SamplingSpace::SO3>();
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

template <SamplingSpace SamplingSpaceType>
Eigen::Vector3d sampleToCloudPos(
    const Eigen::Matrix<double, sampleDim<SamplingSpaceType>(), 1>& sample)
{
  return sample.head(3);
}

template <>
inline Eigen::Vector3d sampleToCloudPos<SamplingSpace::R2>(
    const Eigen::Matrix<double, sampleDim<SamplingSpace::R2>(), 1>& sample)
{
  return Eigen::Vector3d(sample.x(), sample.y(), 0);
}

template <>
inline Eigen::Vector3d sampleToCloudPos<SamplingSpace::SO2>(
    const Eigen::Matrix<double, sampleDim<SamplingSpace::SO2>(), 1>& sample)
{
  return Eigen::Vector3d(sample.x(), 0, 0);
}

template <>
inline Eigen::Vector3d sampleToCloudPos<SamplingSpace::SO3>(
    const Eigen::Matrix<double, sampleDim<SamplingSpace::SO3>(), 1>& sample)
{
  Eigen::AngleAxisd aa(Eigen::Quaterniond(sample.w(), sample.x(), sample.y(), sample.z()));
  return aa.angle() * aa.axis();
}

template <SamplingSpace SamplingSpaceType>
Eigen::Matrix<double, inputDim<SamplingSpaceType>(), 1> sampleToInput(
    const Eigen::Matrix<double, sampleDim<SamplingSpaceType>(), 1>& sample)
{
  mc_rtc::log::error_and_throw<std::runtime_error>(
      "[sampleToInput] Need to be specialized for {}.", std::to_string(SamplingSpaceType));
}

template <>
inline Eigen::Matrix<double, inputDim<SamplingSpace::R2>(), 1> sampleToInput<SamplingSpace::R2>(
    const Eigen::Matrix<double, sampleDim<SamplingSpace::R2>(), 1>& sample)
{
  return sample;
}

template <>
inline Eigen::Matrix<double, inputDim<SamplingSpace::SO2>(), 1> sampleToInput<SamplingSpace::SO2>(
    const Eigen::Matrix<double, sampleDim<SamplingSpace::SO2>(), 1>& sample)
{
  double cos = std::cos(sample.x());
  double sin = std::sin(sample.x());
  Eigen::Matrix<double, inputDim<SamplingSpace::SO2>(), 1> input;
  input << cos, -sin, sin, cos;
  return input;
}

template <>
inline Eigen::Matrix<double, inputDim<SamplingSpace::SE2>(), 1> sampleToInput<SamplingSpace::SE2>(
    const Eigen::Matrix<double, sampleDim<SamplingSpace::SE2>(), 1>& sample)
{
  Eigen::Matrix<double, inputDim<SamplingSpace::SE2>(), 1> input;
  input <<
      sampleToInput<SamplingSpace::R2>(sample.head<sampleDim<SamplingSpace::R2>()>()),
      sampleToInput<SamplingSpace::SO2>(sample.tail<sampleDim<SamplingSpace::SO2>()>());
  return input;
}

template <>
inline Eigen::Matrix<double, inputDim<SamplingSpace::R3>(), 1> sampleToInput<SamplingSpace::R3>(
    const Eigen::Matrix<double, sampleDim<SamplingSpace::R3>(), 1>& sample)
{
  return sample;
}

template <>
inline Eigen::Matrix<double, inputDim<SamplingSpace::SO3>(), 1> sampleToInput<SamplingSpace::SO3>(
    const Eigen::Matrix<double, sampleDim<SamplingSpace::SO3>(), 1>& sample)
{
  Eigen::Quaterniond quat(sample.w(), sample.x(), sample.y(), sample.z());
  Eigen::Matrix3d mat = quat.toRotationMatrix();
  // Arrange the elements row by row
  // https://stackoverflow.com/a/22896750
  mat.transposeInPlace();
  return Eigen::Map<Eigen::VectorXd>(mat.data(), mat.size());
}

template <>
inline Eigen::Matrix<double, inputDim<SamplingSpace::SE3>(), 1> sampleToInput<SamplingSpace::SE3>(
    const Eigen::Matrix<double, sampleDim<SamplingSpace::SE3>(), 1>& sample)
{
  Eigen::Matrix<double, inputDim<SamplingSpace::SE3>(), 1> input;
  input <<
      sampleToInput<SamplingSpace::R3>(sample.head<sampleDim<SamplingSpace::R3>()>()),
      sampleToInput<SamplingSpace::SO3>(sample.tail<sampleDim<SamplingSpace::SO3>()>());
  return input;
}
}
