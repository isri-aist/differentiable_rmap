/* Author: Masaki Murooka */

#include <optmotiongen_msgs/RobotStateArray.h>

#include <optmotiongen/Utils/RosUtils.h>

#include <differentiable_rmap/RmapSamplingFootstep.h>

using namespace DiffRmap;


template <SamplingSpace SamplingSpaceType>
RmapSamplingFootstep<SamplingSpaceType>::RmapSamplingFootstep(
    const std::shared_ptr<OmgCore::Robot>& rb,
    const std::string& support_foot_body_name,
    const std::string& swing_foot_body_name,
    const std::string& waist_body_name):
    RmapSamplingIK<SamplingSpaceType>(rb),
    support_foot_body_name_(support_foot_body_name),
    swing_foot_body_name_(swing_foot_body_name),
    waist_body_name_(waist_body_name),
    support_foot_body_idx_(rb->bodyIndexByName(support_foot_body_name_)),
  swing_foot_body_idx_(rb->bodyIndexByName(swing_foot_body_name_)),
  waist_body_idx_(rb->bodyIndexByName(waist_body_name_))
{
  // Setup task and problem
  support_foot_body_task_ = std::make_shared<OmgCore::BodyPoseTask>(
      std::make_shared<OmgCore::BodyFunc>(
          rb_arr_,
          0,
          support_foot_body_name_),
      sva::PTransformd::Identity(),
      "SupportFootBodyPoseTask");
  swing_foot_body_task_ = std::make_shared<OmgCore::BodyPoseTask>(
      std::make_shared<OmgCore::BodyFunc>(
          rb_arr_,
          0,
          swing_foot_body_name_),
      sva::PTransformd::Identity(),
      "SwingFootBodyPoseTask");
  waist_body_task_ = std::make_shared<OmgCore::BodyPoseTask>(
      std::make_shared<OmgCore::BodyFunc>(
          rb_arr_,
          0,
          waist_body_name_),
      sva::PTransformd::Identity(),
      "WaistBodyPoseTask");

  taskset_.addTask(support_foot_body_task_);
  taskset_.addTask(swing_foot_body_task_);
  taskset_.addTask(waist_body_task_);

  problem_ = std::make_shared<OmgCore::IterativeQpProblem>(rb_arr_);
  problem_->setup(
      std::vector<OmgCore::Taskset>{taskset_},
      std::vector<OmgCore::QpSolverType>{OmgCore::QpSolverType::JRLQP});

  // Copy problem rbc_arr to member rb_arr to synchronize them
  rbc_arr_ = problem_->rbcArr();
}

template <SamplingSpace SamplingSpaceType>
void RmapSamplingFootstep<SamplingSpaceType>::configure(const mc_rtc::Configuration& mc_rtc_config)
{
  RmapSamplingIK<SamplingSpaceType>::configure(mc_rtc_config);
  config_.load(mc_rtc_config);
}

template <SamplingSpace SamplingSpaceType>
void RmapSamplingFootstep<SamplingSpaceType>::setupSampling()
{
  // Calculate coefficient and offset to make random position
  footstep_pos_coeff_ = (config_.upper_footstep_pos - config_.lower_footstep_pos) / 2;
  footstep_pos_offset_ = (config_.upper_footstep_pos + config_.lower_footstep_pos) / 2;
}

template <SamplingSpace SamplingSpaceType>
void RmapSamplingFootstep<SamplingSpaceType>::sampleOnce(int sample_idx)
{
  // Set IK target of foot
  support_foot_body_task_->target() = sva::PTransformd::Identity();
  const FootstepPos& footstep_pos = footstep_pos_coeff_.cwiseProduct(FootstepPos::Random()) + footstep_pos_offset_;
  swing_foot_body_task_->target().translation().head<2>() = footstep_pos.head<2>();
  swing_foot_body_task_->target().translation().z() = 0;
  swing_foot_body_task_->target().rotation() = Eigen::AngleAxisd(
      footstep_pos.z(), Eigen::Vector3d::UnitZ()).toRotationMatrix().transpose();

  bool reachability = true;

  for (int i = 0; i < 3; i++) {
    const auto& rb = rb_arr_[0];
    const auto& rbc = rbc_arr_[0];

    // Set IK target of waist
    switch (i) {
      case 0:
        waist_body_task_->target() = support_foot_body_task_->target();
        break;
      case 1:
        waist_body_task_->target() = swing_foot_body_task_->target();
        break;
      case 2:
        waist_body_task_->target() = sva::interpolate(
            support_foot_body_task_->target(),
            swing_foot_body_task_->target(),
            0.5);
        break;
    }
    waist_body_task_->target().translation().z() = config_.waist_height;

    // Set zero configuration
    rbc->zero(*rb);
    rbd::forwardKinematics(*rb, *rbc);

    // Solve IK
    problem_->run(config_.ik_loop_num);
    taskset_.update(rb_arr_, rbc_arr_, aux_rb_arr_);

    if (taskset_.errorSquaredNorm() > std::pow(config_.ik_error_thre, 2)) {
      reachability = false;
      break;
    }
  }

  // Append new sample to sample list
  const SampleVector& sample = poseToSample<SamplingSpaceType>(swing_foot_body_task_->target());
  sample_list_[sample_idx] = sample;
  reachability_list_[sample_idx] = reachability;
  if (reachability) {
    reachable_cloud_msg_.points.push_back(OmgCore::toPoint32Msg(sampleToCloudPos<SamplingSpaceType>(sample)));
  } else {
    unreachable_cloud_msg_.points.push_back(OmgCore::toPoint32Msg(sampleToCloudPos<SamplingSpaceType>(sample)));
  }
}

std::shared_ptr<RmapSamplingBase> DiffRmap::createRmapSamplingFootstep(
    SamplingSpace sampling_space,
    const std::shared_ptr<OmgCore::Robot>& rb,
    const std::string& support_foot_body_name,
    const std::string& swing_foot_body_name,
    const std::string& waist_body_name)
{
  if (sampling_space == SamplingSpace::R2) {
    return std::make_shared<RmapSamplingFootstep<SamplingSpace::R2>>(
        rb, support_foot_body_name, swing_foot_body_name, waist_body_name);
  } else if (sampling_space == SamplingSpace::SO2) {
    return std::make_shared<RmapSamplingFootstep<SamplingSpace::SO2>>(
        rb, support_foot_body_name, swing_foot_body_name, waist_body_name);
  } else if (sampling_space == SamplingSpace::SE2) {
    return std::make_shared<RmapSamplingFootstep<SamplingSpace::SE2>>(
        rb, support_foot_body_name, swing_foot_body_name, waist_body_name);
  } else if (sampling_space == SamplingSpace::R3) {
    return std::make_shared<RmapSamplingFootstep<SamplingSpace::R3>>(
        rb, support_foot_body_name, swing_foot_body_name, waist_body_name);
  } else if (sampling_space == SamplingSpace::SO3) {
    return std::make_shared<RmapSamplingFootstep<SamplingSpace::SO3>>(
        rb, support_foot_body_name, swing_foot_body_name, waist_body_name);
  } else if (sampling_space == SamplingSpace::SE3) {
    return std::make_shared<RmapSamplingFootstep<SamplingSpace::SE3>>(
        rb, support_foot_body_name, swing_foot_body_name, waist_body_name);
  } else {
    mc_rtc::log::error_and_throw<std::runtime_error>(
        "[createRmapSamplingFootstep] Unsupported SamplingSpace: {}", std::to_string(sampling_space));
  }
}

// Declare template specialized class
// See https://stackoverflow.com/a/8752879
template class RmapSamplingFootstep<SamplingSpace::R2>;
template class RmapSamplingFootstep<SamplingSpace::SO2>;
template class RmapSamplingFootstep<SamplingSpace::SE2>;
template class RmapSamplingFootstep<SamplingSpace::R3>;
template class RmapSamplingFootstep<SamplingSpace::SO3>;
template class RmapSamplingFootstep<SamplingSpace::SE3>;
