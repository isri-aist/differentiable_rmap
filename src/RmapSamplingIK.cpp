/* Author: Masaki Murooka */

#include <optmotiongen_msgs/RobotStateArray.h>

#include <optmotiongen/Utils/RosUtils.h>

#include <differentiable_rmap/RmapSamplingIK.h>

using namespace DiffRmap;


template <SamplingSpace SamplingSpaceType>
RmapSamplingIK<SamplingSpaceType>::RmapSamplingIK(
    const std::shared_ptr<OmgCore::Robot>& rb,
    const std::string& body_name,
    const std::vector<std::string>& joint_name_list):
    RmapSampling<SamplingSpaceType>(rb, body_name, joint_name_list)
{
}

template <SamplingSpace SamplingSpaceType>
void RmapSamplingIK<SamplingSpaceType>::sampleOnce(int sample_idx)
{
  const auto& rb = rb_arr_[0];
  const auto& rbc = rbc_arr_[0];

  Eigen::VectorXd joint_pos =
      joint_pos_coeff_.cwiseProduct(Eigen::VectorXd::Random(joint_name_list_.size())) + joint_pos_offset_;
  for (size_t i = 0; i < joint_name_list_.size(); i++) {
    rbc->q[joint_idx_list_[i]][0] = joint_pos[i];
  }
  rbd::forwardKinematics(*rb, *rbc);
  const auto& body_pose = rbc->bodyPosW[body_idx_];
  const SampleVector& sample = poseToSample<SamplingSpaceType>(body_pose);
  sample_list_[sample_idx] = sample;
  reachability_list_[sample_idx] = true;
  reachable_cloud_msg_.points.push_back(OmgCore::toPoint32Msg(sampleToCloudPos<SamplingSpaceType>(sample)));

  ROS_INFO("tmp RmapSamplingIK.");
}

std::shared_ptr<RmapSamplingBase> DiffRmap::createRmapSamplingIK(
    SamplingSpace sampling_space,
    const std::shared_ptr<OmgCore::Robot>& rb,
    const std::string& body_name,
    const std::vector<std::string>& joint_name_list)
{
  if (sampling_space == SamplingSpace::R2) {
    return std::make_shared<RmapSamplingIK<SamplingSpace::R2>>(rb, body_name, joint_name_list);
  } else if (sampling_space == SamplingSpace::SO2) {
    return std::make_shared<RmapSamplingIK<SamplingSpace::SO2>>(rb, body_name, joint_name_list);
  } else if (sampling_space == SamplingSpace::SE2) {
    return std::make_shared<RmapSamplingIK<SamplingSpace::SE2>>(rb, body_name, joint_name_list);
  } else if (sampling_space == SamplingSpace::R3) {
    return std::make_shared<RmapSamplingIK<SamplingSpace::R3>>(rb, body_name, joint_name_list);
  } else if (sampling_space == SamplingSpace::SO3) {
    return std::make_shared<RmapSamplingIK<SamplingSpace::SO3>>(rb, body_name, joint_name_list);
  } else if (sampling_space == SamplingSpace::SE3) {
    return std::make_shared<RmapSamplingIK<SamplingSpace::SE3>>(rb, body_name, joint_name_list);
  } else {
    mc_rtc::log::error_and_throw<std::runtime_error>(
        "[createRmapSamplingIK] Unsupported SamplingSpace: {}", std::to_string(sampling_space));
  }
}
