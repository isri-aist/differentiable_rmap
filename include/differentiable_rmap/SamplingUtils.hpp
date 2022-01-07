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
Sample<SamplingSpaceType> poseToSample(const sva::PTransformd& pose)
{
  mc_rtc::log::error_and_throw<std::runtime_error>(
      "[poseToSample] Need to be specialized for {}.", std::to_string(SamplingSpaceType));
}

template <>
inline Sample<SamplingSpace::R2> poseToSample<SamplingSpace::R2>(
    const sva::PTransformd& pose)
{
  return pose.translation().head(2);
}

template <>
inline Sample<SamplingSpace::SO2> poseToSample<SamplingSpace::SO2>(
    const sva::PTransformd& pose)
{
  Sample<SamplingSpace::SO2> sample;
  sample << calcYawAngle(pose.rotation().transpose());
  return sample;
}

template <>
inline Sample<SamplingSpace::SE2> poseToSample<SamplingSpace::SE2>(
    const sva::PTransformd& pose)
{
  Sample<SamplingSpace::SE2> sample;
  sample << poseToSample<SamplingSpace::R2>(pose), poseToSample<SamplingSpace::SO2>(pose);
  return sample;
}

template <>
inline Sample<SamplingSpace::R3> poseToSample<SamplingSpace::R3>(
    const sva::PTransformd& pose)
{
  return pose.translation();
}

template <>
inline Sample<SamplingSpace::SO3> poseToSample<SamplingSpace::SO3>(
    const sva::PTransformd& pose)
{
  // Element order is (x, y, z, w)
  Sample<SamplingSpace::SO3> sample;
  sample << Eigen::Quaterniond(pose.rotation().transpose()).coeffs();
  return sample;
}

template <>
inline Sample<SamplingSpace::SE3> poseToSample<SamplingSpace::SE3>(
    const sva::PTransformd& pose)
{
  Sample<SamplingSpace::SE3> sample;
  sample << poseToSample<SamplingSpace::R3>(pose), poseToSample<SamplingSpace::SO3>(pose);
  return sample;
}

template <SamplingSpace SamplingSpaceType>
sva::PTransformd sampleToPose(const Sample<SamplingSpaceType>& sample)
{
  mc_rtc::log::error_and_throw<std::runtime_error>(
      "[sampleToPose] Need to be specialized for {}.", std::to_string(SamplingSpaceType));
}

template <>
inline sva::PTransformd sampleToPose<SamplingSpace::R2>(
    const Sample<SamplingSpace::R2>& sample)
{
  return sva::PTransformd(Eigen::Vector3d(sample.x(), sample.y(), 0));
}

template <>
inline sva::PTransformd sampleToPose<SamplingSpace::SO2>(
    const Sample<SamplingSpace::SO2>& sample)
{
  return sva::PTransformd(Eigen::Matrix3d(
      Eigen::AngleAxisd(sample.x(), Eigen::Vector3d::UnitZ()).toRotationMatrix().transpose()));
}

template <>
inline sva::PTransformd sampleToPose<SamplingSpace::SE2>(
    const Sample<SamplingSpace::SE2>& sample)
{
  return sva::PTransformd(
      Eigen::AngleAxisd(sample.z(), Eigen::Vector3d::UnitZ()).toRotationMatrix().transpose(),
      Eigen::Vector3d(sample.x(), sample.y(), 0));
}

template <>
inline sva::PTransformd sampleToPose<SamplingSpace::R3>(
    const Sample<SamplingSpace::R3>& sample)
{
  return sva::PTransformd(sample);
}

template <>
inline sva::PTransformd sampleToPose<SamplingSpace::SO3>(
    const Sample<SamplingSpace::SO3>& sample)
{
  return sva::PTransformd(Eigen::Quaterniond(sample.w(), sample.x(), sample.y(), sample.z()).inverse());
}

template <>
inline sva::PTransformd sampleToPose<SamplingSpace::SE3>(
    const Sample<SamplingSpace::SE3>& sample)
{
  return sva::PTransformd(
      Eigen::Quaterniond(sample.tail<sampleDim<SamplingSpace::SO3>()>().w(),
                         sample.tail<sampleDim<SamplingSpace::SO3>()>().x(),
                         sample.tail<sampleDim<SamplingSpace::SO3>()>().y(),
                         sample.tail<sampleDim<SamplingSpace::SO3>()>().z()).inverse(),
      sample.head<sampleDim<SamplingSpace::R3>()>());
}

template <SamplingSpace SamplingSpaceType>
Eigen::Vector3d sampleToCloudPos(const Sample<SamplingSpaceType>& sample)
{
  return sample.head(3);
}

template <>
inline Eigen::Vector3d sampleToCloudPos<SamplingSpace::R2>(
    const Sample<SamplingSpace::R2>& sample)
{
  return Eigen::Vector3d(sample.x(), sample.y(), 0);
}

template <>
inline Eigen::Vector3d sampleToCloudPos<SamplingSpace::SO2>(
    const Sample<SamplingSpace::SO2>& sample)
{
  return Eigen::Vector3d(sample.x(), 0, 0);
}

template <>
inline Eigen::Vector3d sampleToCloudPos<SamplingSpace::SO3>(
    const Sample<SamplingSpace::SO3>& sample)
{
  Eigen::AngleAxisd aa(Eigen::Quaterniond(sample.w(), sample.x(), sample.y(), sample.z()));
  return aa.angle() * aa.axis();
}

template <SamplingSpace SamplingSpaceType>
Input<SamplingSpaceType> sampleToInput(const Sample<SamplingSpaceType>& sample)
{
  mc_rtc::log::error_and_throw<std::runtime_error>(
      "[sampleToInput] Need to be specialized for {}.", std::to_string(SamplingSpaceType));
}

template <>
inline Input<SamplingSpace::R2> sampleToInput<SamplingSpace::R2>(
    const Sample<SamplingSpace::R2>& sample)
{
  return sample;
}

template <>
inline Input<SamplingSpace::SO2> sampleToInput<SamplingSpace::SO2>(
    const Sample<SamplingSpace::SO2>& sample)
{
  double cos = std::cos(sample.x());
  double sin = std::sin(sample.x());
  Input<SamplingSpace::SO2> input;
  input << cos, -sin, sin, cos;
  return input;
}

template <>
inline Input<SamplingSpace::SE2> sampleToInput<SamplingSpace::SE2>(
    const Sample<SamplingSpace::SE2>& sample)
{
  Input<SamplingSpace::SE2> input;
  input <<
      sampleToInput<SamplingSpace::R2>(sample.head<sampleDim<SamplingSpace::R2>()>()),
      sampleToInput<SamplingSpace::SO2>(sample.tail<sampleDim<SamplingSpace::SO2>()>());
  return input;
}

template <>
inline Input<SamplingSpace::R3> sampleToInput<SamplingSpace::R3>(
    const Sample<SamplingSpace::R3>& sample)
{
  return sample;
}

template <>
inline Input<SamplingSpace::SO3> sampleToInput<SamplingSpace::SO3>(
    const Sample<SamplingSpace::SO3>& sample)
{
  Eigen::Quaterniond quat(sample.w(), sample.x(), sample.y(), sample.z());
  Eigen::Matrix3d mat = quat.toRotationMatrix();
  // Arrange the elements row by row
  // https://stackoverflow.com/a/22896750
  mat.transposeInPlace();
  return Eigen::Map<Eigen::VectorXd>(mat.data(), mat.size());
}

template <>
inline Input<SamplingSpace::SE3> sampleToInput<SamplingSpace::SE3>(
    const Sample<SamplingSpace::SE3>& sample)
{
  Input<SamplingSpace::SE3> input;
  input <<
      sampleToInput<SamplingSpace::R3>(sample.head<sampleDim<SamplingSpace::R3>()>()),
      sampleToInput<SamplingSpace::SO3>(sample.tail<sampleDim<SamplingSpace::SO3>()>());
  return input;
}
}
