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

  if constexpr (SamplingSpaceType == SamplingSpace::R2 ||
                SamplingSpaceType == SamplingSpace::SO2 ||
                SamplingSpaceType == SamplingSpace::SE2) {
      body_task_->target().translation().head<2>().setRandom();
      body_task_->target().translation().z() = 0;
      body_task_->target().rotation() = Eigen::AngleAxisd(
          M_PI * Eigen::Matrix<double, 1, 1>::Random()[0],
          Eigen::Vector3d::UnitZ()).toRotationMatrix();
    } else {
    body_task_->target().translation().setRandom();
    body_task_->target().rotation() = Eigen::Quaterniond::UnitRandom().toRotationMatrix();
  }
  Eigen::Vector3d body_pos = 2 * Eigen::Vector3d::Random();
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
