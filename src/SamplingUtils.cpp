/* Author: Masaki Murooka */

#include <SpaceVecAlg/SpaceVecAlg>
#include <mc_rtc/logging.h>

#include <differentiable_rmap/SamplingUtils.h>
#include <differentiable_rmap/MathUtils.h>

using namespace DiffRmap;


SamplingSpace DiffRmap::strToSamplingSpace(const std::string& sampling_space_str)
{
  if (sampling_space_str == "R2") {
    return SamplingSpace::R2;
  } else if (sampling_space_str == "SO2") {
    return SamplingSpace::SO2;
  } else if (sampling_space_str == "SE2") {
    return SamplingSpace::SE2;
  } else if (sampling_space_str == "R3") {
    return SamplingSpace::R3;
  } else if (sampling_space_str == "SO3") {
    return SamplingSpace::SO3;
  } else if (sampling_space_str == "SE3") {
    return SamplingSpace::SE3;
  } else {
    mc_rtc::log::error_and_throw<std::runtime_error>(
        "[strToQpSolverType] Unsupported SamplingSpace name: {}", sampling_space_str);
  }
}

Eigen::VectorXd DiffRmap::poseToSample(const sva::PTransformd& pose,
                                       SamplingSpace sampling_space)
{
  Eigen::VectorXd sample;
  if (sampling_space == SamplingSpace::R2) {
    sample.resize(2);
    sample << pose.translation().head(2);
  } else if (sampling_space == SamplingSpace::SO2) {
    sample.resize(1);
    sample << calcYawAngle(pose.rotation().transpose());
  } else if (sampling_space == SamplingSpace::SE2) {
    Eigen::VectorXd sample_r2 = poseToSample(pose, SamplingSpace::R2);
    Eigen::VectorXd sample_so2 = poseToSample(pose, SamplingSpace::SO2);
    sample.resize(sample_r2.size() + sample_so2.size());
    sample << sample_r2, sample_so2;
  } else if (sampling_space == SamplingSpace::R3) {
    sample.resize(3);
    sample << pose.translation();
  } else if (sampling_space == SamplingSpace::SO3) {
    sample.resize(4);
    // Element order is (x, y, z, w)
    sample << Eigen::Quaterniond(pose.rotation().transpose()).coeffs();
  } else if (sampling_space == SamplingSpace::SE3) {
    Eigen::VectorXd sample_r3 = poseToSample(pose, SamplingSpace::R3);
    Eigen::VectorXd sample_so3 = poseToSample(pose, SamplingSpace::SO3);
    sample.resize(sample_r3.size() + sample_so3.size());
    sample << sample_r3, sample_so3;
  }
  return sample;
}
