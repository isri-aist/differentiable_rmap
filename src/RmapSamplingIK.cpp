/* Author: Masaki Murooka */

#include <optmotiongen_msgs/RobotStateArray.h>

#include <optmotiongen/Utils/RosUtils.h>

#include <differentiable_rmap/RmapSamplingIK.h>

using namespace DiffRmap;


namespace
{
/** \brief Get selection indices of task value depending on sampling space. */
std::vector<size_t> getSelectIdxs(SamplingSpace sampling_space)
{
  switch (sampling_space) {
    case SamplingSpace::R2:
      return std::vector<size_t>{3, 4};
    case SamplingSpace::SO2:
      return std::vector<size_t>{2};
    case SamplingSpace::SE2:
      return std::vector<size_t>{2, 3, 4};
    case SamplingSpace::R3:
      return std::vector<size_t>{3, 4, 5};
    case SamplingSpace::SO3:
      return std::vector<size_t>{0, 1, 2};
    case SamplingSpace::SE3:
      return std::vector<size_t>{0, 1, 2, 3, 4, 5};
    default:
      mc_rtc::log::error_and_throw<std::runtime_error>(
          "[getSelectIdxs] SamplingSpace {} is not supported.", std::to_string(sampling_space));
  }
}
}

template <SamplingSpace SamplingSpaceType>
RmapSamplingIK<SamplingSpaceType>::RmapSamplingIK(
    const std::shared_ptr<OmgCore::Robot>& rb,
    const std::string& body_name,
    const std::vector<std::string>& joint_name_list):
    RmapSampling<SamplingSpaceType>(rb, body_name, joint_name_list)
{
  // Setup task and problem
  body_task_ = std::make_shared<OmgCore::BodyPoseTask>(
      std::make_shared<OmgCore::BodyFunc>(
          rb_arr_,
          0,
          body_name_),
      sva::PTransformd::Identity(),
      "BodyPoseTask",
      getSelectIdxs(SamplingSpaceType));

  taskset_.addTask(body_task_);

  problem_ = std::make_shared<OmgCore::IterativeQpProblem>(rb_arr_);
  problem_->setup(
      std::vector<OmgCore::Taskset>{taskset_},
      std::vector<OmgCore::QpSolverType>{OmgCore::QpSolverType::OSQP});

  // Copy problem rbc_arr to member rb_arr to synchronize them
  rbc_arr_ = problem_->rbcArr();
}

template <SamplingSpace SamplingSpaceType>
void RmapSamplingIK<SamplingSpaceType>::setupSampling()
{
  // Overwrite joint range to restrict joints to be used
  // Be carefull that this overwrites original robot
  // This becomes unnecessary when optmotiongen supports joint selection
  const auto& rb = rb_arr_[0];
  for (const auto& joint : rb->joints()) {
    if (std::find(joint_name_list_.begin(), joint_name_list_.end(), joint.name())
        != joint_name_list_.end()) {
      continue;
    }

    int joint_idx = rb->jointIndexByName(joint.name());
    std::fill(rb->jposs_min_[joint_idx].begin(), rb->jposs_min_[joint_idx].end(), 0);
    std::fill(rb->jposs_max_[joint_idx].begin(), rb->jposs_max_[joint_idx].end(), 0);
  }

  // Get upper and lower position of bounding box in configuration space
  sample_list_.resize(bbox_sample_num_);
  reachability_list_.resize(bbox_sample_num_);

  RmapSampling<SamplingSpaceType>::setupSampling();
  for (int i = 0; i < bbox_sample_num_; i++) {
    RmapSampling<SamplingSpaceType>::sampleOnce(i);
  }

  Eigen::Vector3d upper_body_pos = Eigen::Vector3d::Constant(-1e10);
  Eigen::Vector3d lower_body_pos = Eigen::Vector3d::Constant(1e10);
  for (const SampleVector& sample : sample_list_) {
    const Eigen::Vector3d& cloud_pos = sampleToCloudPos<SamplingSpaceType>(sample);
    upper_body_pos = upper_body_pos.cwiseMax(cloud_pos);
    lower_body_pos = lower_body_pos.cwiseMin(cloud_pos);
  }

  // Calculate coefficient and offset to make random position
  constexpr double bbox_padding_rate = 1.2;
  body_pos_coeff_ = bbox_padding_rate * (upper_body_pos - lower_body_pos) / 2;
  body_pos_offset_ = (upper_body_pos + lower_body_pos) / 2;
}

template <SamplingSpace SamplingSpaceType>
void RmapSamplingIK<SamplingSpaceType>::sampleOnce(int sample_idx)
{
  // Set IK target
  if constexpr (SamplingSpaceType == SamplingSpace::R2 ||
                SamplingSpaceType == SamplingSpace::SO2 ||
                SamplingSpaceType == SamplingSpace::SE2) {
      body_task_->target().translation().head<2>() =
          body_pos_coeff_.head<2>().cwiseProduct(Eigen::Vector2d::Random()) + body_pos_offset_.head<2>();
      body_task_->target().translation().z() = 0;
      body_task_->target().rotation() = Eigen::AngleAxisd(
          M_PI * Eigen::Matrix<double, 1, 1>::Random()[0],
          Eigen::Vector3d::UnitZ()).toRotationMatrix();
    } else {
    body_task_->target().translation() =
        body_pos_coeff_.cwiseProduct(Eigen::Vector3d::Random()) + body_pos_offset_;
    body_task_->target().rotation() = Eigen::Quaterniond::UnitRandom().toRotationMatrix();
  }

  bool reachability = false;

  for (int i = 0; i < ik_trial_num_; i++) {
    const auto& rb = rb_arr_[0];
    const auto& rbc = rbc_arr_[0];

    if (i == 0) {
      // Set zero configuration
      rbc->zero(*rb);
    } else {
      // Set random configuration
      Eigen::VectorXd joint_pos =
          joint_pos_coeff_.cwiseProduct(Eigen::VectorXd::Random(joint_name_list_.size())) + joint_pos_offset_;
      for (size_t i = 0; i < joint_name_list_.size(); i++) {
        rbc->q[joint_idx_list_[i]][0] = joint_pos[i];
      }
    }
    rbd::forwardKinematics(*rb, *rbc);

    // Solve IK
    problem_->run(ik_loop_num_);
    taskset_.update(rb_arr_, rbc_arr_, aux_rb_arr_);

    if (taskset_.errorSquaredNorm() < std::pow(ik_error_thre_, 2)) {
      reachability = true;
      break;
    }
  }

  // Append new sample to sample list
  const SampleVector& sample = poseToSample<SamplingSpaceType>(body_task_->target());
  sample_list_[sample_idx] = sample;
  reachability_list_[sample_idx] = reachability;
  if (reachability) {
    reachable_cloud_msg_.points.push_back(OmgCore::toPoint32Msg(sampleToCloudPos<SamplingSpaceType>(sample)));
  } else {
    unreachable_cloud_msg_.points.push_back(OmgCore::toPoint32Msg(sampleToCloudPos<SamplingSpaceType>(sample)));
  }
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
