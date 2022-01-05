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
  body_task_ = std::make_shared<OmgCore::BodyPoseTask>(
      std::make_shared<OmgCore::BodyFunc>(
          rb_arr_,
          0,
          body_name_),
      sva::PTransformd::Identity(),
      "BodyPoseTask",
      std::vector<size_t>{3, 4});

  taskset_.addTask(body_task_);

  problem_ = std::make_shared<OmgCore::IterativeQpProblem>(rb_arr_);
  problem_->setup(
      std::vector<OmgCore::Taskset>{taskset_},
      std::vector<OmgCore::QpSolverType>{OmgCore::QpSolverType::OSQP});
  problem_->printInfo(true);

  rbc_arr_ = problem_->rbcArr();
}

template <SamplingSpace SamplingSpaceType>
void RmapSamplingIK<SamplingSpaceType>::sampleOnce(int sample_idx)
{
  const auto& rb = rb_arr_[0];
  const auto& rbc = rbc_arr_[0];
  rbc->zero(*rb);
  rb->update(*rbc);

  Eigen::Vector3d body_pos = 2 * Eigen::Vector3d::Random();
  body_task_->target() = sva::PTransformd(body_pos);
  problem_->run(ik_loop_num_);
  taskset_.update(rb_arr_, rbc_arr_, aux_rb_arr_);
  bool reachability = (taskset_.errorSquaredNorm() < std::pow(ik_error_thre_, 2));

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
