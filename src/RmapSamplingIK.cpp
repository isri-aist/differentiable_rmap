/* Author: Masaki Murooka */

#include <optmotiongen_msgs/RobotStateArray.h>

#include <optmotiongen_core/Utils/RosUtils.h>

#include <differentiable_rmap/RmapSamplingIK.h>

using namespace DiffRmap;

namespace
{
/** \brief Get selection indices of task value depending on sampling space. */
std::vector<size_t> getSelectIdxs(SamplingSpace sampling_space)
{
  switch(sampling_space)
  {
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
      mc_rtc::log::error_and_throw<std::runtime_error>("[getSelectIdxs] SamplingSpace {} is not supported.",
                                                       std::to_string(sampling_space));
  }
}
} // namespace

template<SamplingSpace SamplingSpaceType>
RmapSamplingIK<SamplingSpaceType>::RmapSamplingIK(const std::shared_ptr<OmgCore::Robot> & rb)
: RmapSampling<SamplingSpaceType>(rb)
{
}

template<SamplingSpace SamplingSpaceType>
RmapSamplingIK<SamplingSpaceType>::RmapSamplingIK(const std::shared_ptr<OmgCore::Robot> & rb,
                                                  const std::string & body_name,
                                                  const std::vector<std::string> & joint_name_list)
: RmapSampling<SamplingSpaceType>(rb, body_name, joint_name_list)
{
}

template<SamplingSpace SamplingSpaceType>
void RmapSamplingIK<SamplingSpaceType>::configure(const mc_rtc::Configuration & mc_rtc_config)
{
  RmapSampling<SamplingSpaceType>::configure(mc_rtc_config);
  config_.load(mc_rtc_config);
}

template<SamplingSpace SamplingSpaceType>
void RmapSamplingIK<SamplingSpaceType>::setup()
{
  // setupCollisionTask is called from setupSampling()
  setupSampling();
}

template<SamplingSpace SamplingSpaceType>
void RmapSamplingIK<SamplingSpaceType>::setupSampling()
{
  // Eigen's random seed can be set by srand
  srand(config_.random_seed);

  // Set robot root pose
  rb_arr_[0]->rootPose(config_.root_pose);

  // Setup task
  SamplingSpace ik_constraint_space = SamplingSpaceType;
  if(!config_.ik_constraint_space.empty())
  {
    ik_constraint_space = strToSamplingSpace(config_.ik_constraint_space);
  }
  body_task_ = std::make_shared<OmgCore::BodyPoseTask>(
      std::make_shared<OmgCore::BodyFunc>(rb_arr_, 0, body_name_, config_.body_pose_offset),
      sva::PTransformd::Identity(), "BodyPoseTask", getSelectIdxs(ik_constraint_space));

  this->setupCollisionTask();

  // Setup problem
  taskset_.addTask(body_task_);
  for(const auto & collision_task : collision_task_list_)
  {
    taskset_.addTask(collision_task);
  }
  for(const auto & additional_task : additional_task_list_)
  {
    taskset_.addTask(additional_task);
  }

  problem_ = std::make_shared<OmgCore::IterativeQpProblem>(rb_arr_);
  // JRLQP is superior to other QP solvers in terms of computational time and solvability
  problem_->setup(std::vector<OmgCore::Taskset>{taskset_},
                  std::vector<OmgCore::QpSolverType>{OmgCore::QpSolverType::JRLQP});

  // Copy problem rbc_arr to member rb_arr to synchronize them
  rbc_arr_ = problem_->rbcArr();

  // Overwrite joint range to restrict joints to be used
  // Be carefull that this overwrites original robot
  // This becomes unnecessary when optmotiongen supports joint selection
  const auto & rb = rb_arr_[0];
  for(const auto & joint : rb->joints())
  {
    if(std::find(joint_name_list_.begin(), joint_name_list_.end(), joint.name()) != joint_name_list_.end())
    {
      continue;
    }

    int joint_idx = rb->jointIndexByName(joint.name());
    std::fill(rb->jposs_min_[joint_idx].begin(), rb->jposs_min_[joint_idx].end(), 0);
    std::fill(rb->jposs_max_[joint_idx].begin(), rb->jposs_max_[joint_idx].end(), 0);
  }

  // Get upper and lower position of bounding box in configuration space
  sample_list_.resize(config_.bbox_sample_num);
  reachability_list_.resize(config_.bbox_sample_num);

  RmapSampling<SamplingSpaceType>::setupSampling();
  for(int i = 0; i < config_.bbox_sample_num; i++)
  {
    RmapSampling<SamplingSpaceType>::sampleOnce(i);
  }

  Eigen::Vector3d upper_body_pos = Eigen::Vector3d::Constant(-1e10);
  Eigen::Vector3d lower_body_pos = Eigen::Vector3d::Constant(1e10);
  for(const SampleType & sample : sample_list_)
  {
    const Eigen::Vector3d & cloud_pos = sampleToCloudPos<SamplingSpaceType>(sample);
    upper_body_pos = upper_body_pos.cwiseMax(cloud_pos);
    lower_body_pos = lower_body_pos.cwiseMin(cloud_pos);
  }

  // Calculate coefficient and offset to make random position
  body_pos_coeff_ = config_.bbox_padding_rate * (upper_body_pos - lower_body_pos) / 2;
  body_pos_offset_ = (upper_body_pos + lower_body_pos) / 2;
  body_yaw_coeff_ = (config_.body_yaw_limits.second - config_.body_yaw_limits.first) / 2;
  body_yaw_offset_ = (config_.body_yaw_limits.second + config_.body_yaw_limits.first) / 2;

  reachable_sample_num_ = 0;
}

template<SamplingSpace SamplingSpaceType>
bool RmapSamplingIK<SamplingSpaceType>::sampleOnce(int sample_idx)
{
  // Set IK target
  if constexpr(SamplingSpaceType == SamplingSpace::R2 || SamplingSpaceType == SamplingSpace::SO2
               || SamplingSpaceType == SamplingSpace::SE2)
  {
    body_task_->target().translation().head<2>() =
        body_pos_coeff_.head<2>().cwiseProduct(Eigen::Vector2d::Random()) + body_pos_offset_.head<2>();
    body_task_->target().translation().z() = 0;
    body_task_->target().rotation() =
        Eigen::AngleAxisd(body_yaw_coeff_ * Eigen::Matrix<double, 1, 1>::Random()[0] + body_yaw_offset_,
                          Eigen::Vector3d::UnitZ())
            .toRotationMatrix();
  }
  else
  {
    body_task_->target().translation() = body_pos_coeff_.cwiseProduct(Eigen::Vector3d::Random()) + body_pos_offset_;
    body_task_->target().rotation() = Eigen::Quaterniond::UnitRandom().toRotationMatrix();
  }

  bool reachability = false;

  for(int i = 0; i < config_.ik_trial_num; i++)
  {
    const auto & rb = rb_arr_[0];
    const auto & rbc = rbc_arr_[0];

    if(i == 0)
    {
      // Set zero configuration
      rbc->zero(*rb);
    }
    else
    {
      // Set random configuration
      Eigen::VectorXd joint_pos =
          joint_pos_coeff_.cwiseProduct(Eigen::VectorXd::Random(joint_name_list_.size())) + joint_pos_offset_;
      for(size_t j = 0; j < joint_name_list_.size(); j++)
      {
        rbc->q[joint_idx_list_[j]][0] = joint_pos[j];
      }
    }
    rbd::forwardKinematics(*rb, *rbc);

    // Solve IK
    problem_->run(config_.ik_loop_num);
    taskset_.update(rb_arr_, rbc_arr_, aux_rb_arr_);

    if(taskset_.errorSquaredNorm(false) < std::pow(config_.ik_error_thre, 2))
    {
      reachability = true;
      break;
    }
  }

  // Check ratio of reachable samples
  if(sample_idx > 0)
  {
    double reachable_sample_ratio = static_cast<double>(reachable_sample_num_) / (sample_idx - 1);
    if((reachable_sample_ratio < config_.reachable_sample_ratio_limits.first && !reachability)
       || (reachable_sample_ratio > config_.reachable_sample_ratio_limits.second && reachability))
    {
      return false;
    }
  }

  // Append new sample to sample list
  const SampleType & sample = poseToSample<SamplingSpaceType>(body_task_->target());
  sample_list_[sample_idx] = sample;
  reachability_list_[sample_idx] = reachability;
  if(reachability)
  {
    reachable_sample_num_++;
    reachable_cloud_msg_.points.push_back(OmgCore::toPoint32Msg(sampleToCloudPos<SamplingSpaceType>(sample)));
  }
  else
  {
    unreachable_cloud_msg_.points.push_back(OmgCore::toPoint32Msg(sampleToCloudPos<SamplingSpaceType>(sample)));
  }

  return true;
}

template<SamplingSpace SamplingSpaceType>
void RmapSamplingIK<SamplingSpaceType>::publish()
{
  // Publish robot
  rs_arr_pub_.publish(rb_arr_.makeRobotStateArrayMsg(rbc_arr_));

  // Publish cloud
  {
    const auto & time_now = ros::Time::now();
    reachable_cloud_msg_.header.frame_id = "world";
    reachable_cloud_msg_.header.stamp = time_now;
    reachable_cloud_pub_.publish(reachable_cloud_msg_);
    unreachable_cloud_msg_.header.frame_id = "world";
    unreachable_cloud_msg_.header.stamp = time_now;
    unreachable_cloud_pub_.publish(unreachable_cloud_msg_);
  }

  // Publish collision marker
  std::vector<std::shared_ptr<OmgCore::CollisionTask>> collision_task_list = collision_task_list_;
  for(const auto & task : additional_task_list_)
  {
    if(auto collision_task = std::dynamic_pointer_cast<OmgCore::CollisionTask>(task))
    {
      collision_task_list.push_back(collision_task);
    }
  }
  this->publishCollisionMarker(collision_task_list);
}

std::shared_ptr<RmapSamplingBase> DiffRmap::createRmapSamplingIK(SamplingSpace sampling_space,
                                                                 const std::shared_ptr<OmgCore::Robot> & rb,
                                                                 const std::string & body_name,
                                                                 const std::vector<std::string> & joint_name_list)
{
  if(sampling_space == SamplingSpace::R2)
  {
    return std::make_shared<RmapSamplingIK<SamplingSpace::R2>>(rb, body_name, joint_name_list);
  }
  else if(sampling_space == SamplingSpace::SO2)
  {
    return std::make_shared<RmapSamplingIK<SamplingSpace::SO2>>(rb, body_name, joint_name_list);
  }
  else if(sampling_space == SamplingSpace::SE2)
  {
    return std::make_shared<RmapSamplingIK<SamplingSpace::SE2>>(rb, body_name, joint_name_list);
  }
  else if(sampling_space == SamplingSpace::R3)
  {
    return std::make_shared<RmapSamplingIK<SamplingSpace::R3>>(rb, body_name, joint_name_list);
  }
  else if(sampling_space == SamplingSpace::SO3)
  {
    return std::make_shared<RmapSamplingIK<SamplingSpace::SO3>>(rb, body_name, joint_name_list);
  }
  else if(sampling_space == SamplingSpace::SE3)
  {
    return std::make_shared<RmapSamplingIK<SamplingSpace::SE3>>(rb, body_name, joint_name_list);
  }
  else
  {
    mc_rtc::log::error_and_throw<std::runtime_error>("[createRmapSamplingIK] Unsupported SamplingSpace: {}",
                                                     std::to_string(sampling_space));
  }
}

// Declare template specialized class
// See https://stackoverflow.com/a/8752879
template class RmapSamplingIK<SamplingSpace::R2>;
template class RmapSamplingIK<SamplingSpace::SO2>;
template class RmapSamplingIK<SamplingSpace::SE2>;
template class RmapSamplingIK<SamplingSpace::R3>;
template class RmapSamplingIK<SamplingSpace::SO3>;
template class RmapSamplingIK<SamplingSpace::SE3>;
