/* Author: Masaki Murooka */

#include <numeric>
#include <limits>
#include <chrono>

#include <mc_rtc/constants.h>

#include <geometry_msgs/PoseArray.h>
#include <visualization_msgs/MarkerArray.h>
#include <optmotiongen_msgs/RobotStateArray.h>

#include <optmotiongen/Utils/RosUtils.h>

#include <differentiable_rmap/RmapPlanningPlacement.h>
#include <differentiable_rmap/SVMUtils.h>
#include <differentiable_rmap/GridUtils.h>
#include <differentiable_rmap/libsvm_hotfix.h>

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
RmapPlanningPlacement<SamplingSpaceType>::RmapPlanningPlacement(
    const std::string& svm_path,
    const std::string& bag_path):
    RmapPlanning<SamplingSpaceType>(svm_path, bag_path)
{
  current_pose_arr_pub_ = nh_.template advertise<geometry_msgs::PoseArray>(
      "current_pose_arr", 1, true);
  rs_arr_pub_ = nh_.template advertise<optmotiongen_msgs::RobotStateArray>(
      "robot_state_arr", 1, true);
  posture_srv_ = nh_.advertiseService(
      "generate_posture",
      &RmapPlanningPlacement<SamplingSpaceType>::postureCallback,
      this);
  animate_srv_ = nh_.advertiseService(
      "animate",
      &RmapPlanningPlacement<SamplingSpaceType>::animateCallback,
      this);
}

template <SamplingSpace SamplingSpaceType>
RmapPlanningPlacement<SamplingSpaceType>::~RmapPlanningPlacement()
{
}

template <SamplingSpace SamplingSpaceType>
void RmapPlanningPlacement<SamplingSpaceType>::configure(
    const mc_rtc::Configuration& mc_rtc_config)
{
  RmapPlanning<SamplingSpaceType>::configure(mc_rtc_config);

  config_.load(mc_rtc_config);
}

template <SamplingSpace SamplingSpaceType>
void RmapPlanningPlacement<SamplingSpaceType>::setup(
    const std::shared_ptr<OmgCore::Robot>& rb)
{
  // Setup robot, task, and problem
  if (rb) {
    rb_arr_.clear();
    rb_arr_.push_back(rb);
    rb_arr_.setup();

    body_task_ = std::make_shared<OmgCore::BodyPoseTask>(
        std::make_shared<OmgCore::BodyFunc>(
            rb_arr_,
            0,
            config_.ik_body_name),
        sva::PTransformd::Identity(),
        "BodyPoseTask",
        getSelectIdxs(SamplingSpaceType));
    taskset_.addTask(body_task_);

    problem_ = std::make_shared<OmgCore::IterativeQpProblem>(rb_arr_);
    problem_->setup(
        std::vector<OmgCore::Taskset>{taskset_},
        std::vector<OmgCore::QpSolverType>{OmgCore::QpSolverType::JRLQP});
  }

  // Setup QP coefficients and solver
  int config_dim = placement_vel_dim_ + config_.reaching_num * vel_dim_;
  int svm_ineq_dim = config_.reaching_num;
  int collision_ineq_dim = 0;
  // Introduce variables for inequality constraint errors
  qp_coeff_.setup(
      config_dim + svm_ineq_dim + collision_ineq_dim,
      0,
      svm_ineq_dim + collision_ineq_dim);
  qp_coeff_.x_min_.head(config_dim).setConstant(-config_.delta_config_limit);
  qp_coeff_.x_max_.head(config_dim).setConstant(config_.delta_config_limit);
  qp_coeff_.x_min_.tail(svm_ineq_dim + collision_ineq_dim).setConstant(-1e10);
  qp_coeff_.x_max_.tail(svm_ineq_dim + collision_ineq_dim).setConstant(1e10);

  qp_solver_ = OmgCore::allocateQpSolver(OmgCore::QpSolverType::JRLQP);

  // Setup current and target samples
  current_placement_sample_ = identity_placement_sample_;
  current_reaching_sample_list_.assign(config_.reaching_num, identity_sample_);
  target_reaching_sample_list_.assign(config_.reaching_num, identity_sample_);
}

template <SamplingSpace SamplingSpaceType>
void RmapPlanningPlacement<SamplingSpaceType>::runOnce(bool publish)
{
  int config_dim = placement_vel_dim_ + config_.reaching_num * vel_dim_;
  int svm_ineq_dim = config_.reaching_num;
  int collision_ineq_dim = 0;

  // Set QP objective matrices
  qp_coeff_.obj_mat_.setZero();
  qp_coeff_.obj_vec_.setZero();
  qp_coeff_.obj_mat_.diagonal().template head<placement_vel_dim_>().setConstant(config_.placement_weight);
  qp_coeff_.obj_vec_.template head<placement_vel_dim_>() =
      config_.placement_weight * sampleError<SamplingSpaceType>(target_placement_sample_, current_placement_sample_);
  for (int i = 0; i < config_.reaching_num; i++) {
    int row_idx = placement_vel_dim_ + i * vel_dim_;
    qp_coeff_.obj_mat_.diagonal().template segment<vel_dim_>(row_idx).setConstant(1.0);
    qp_coeff_.obj_vec_.template segment<vel_dim_>(row_idx) =
        sampleError<SamplingSpaceType>(target_reaching_sample_list_[i], current_reaching_sample_list_[i]);
  }
  qp_coeff_.obj_mat_.diagonal().head(config_dim).array() +=
      qp_coeff_.obj_vec_.head(config_dim).squaredNorm() + config_.reg_weight;
  qp_coeff_.obj_mat_.diagonal().tail(svm_ineq_dim + collision_ineq_dim).head(
      svm_ineq_dim).setConstant(config_.svm_ineq_weight);
  // qp_coeff_.obj_mat_.diagonal().tail(svm_ineq_dim + collision_ineq_dim).tail(
  //     collision_ineq_dim).setConstant(config_.collision_ineq_weight);

  // Set QP inequality matrices of reachability
  qp_coeff_.ineq_mat_.setZero();
  qp_coeff_.ineq_vec_.setZero();
  for (int i = 0; i < config_.reaching_num; i++) {
    const PlacementSampleType& pre_sample = current_placement_sample_;
    const SampleType& suc_sample = current_reaching_sample_list_[i];
    const SampleType& rel_sample = relSample<SamplingSpaceType>(pre_sample, suc_sample);
    const VelType& svm_grad = calcSVMGrad<SamplingSpaceType>(
            rel_sample, svm_mo_->param, svm_mo_, svm_coeff_vec_, svm_sv_mat_);
    const VelToVelMat<SamplingSpaceType>& rel_vel_mat_pre =
        relVelToVelMat<SamplingSpaceType>(pre_sample, suc_sample, false);
    const VelToVelMat<SamplingSpaceType>& rel_vel_mat_suc =
        relVelToVelMat<SamplingSpaceType>(pre_sample, suc_sample, true);
    qp_coeff_.ineq_mat_.template block<1, placement_vel_dim_>(i, 0) =
        -1 * svm_grad.transpose() * rel_vel_mat_pre;
    qp_coeff_.ineq_mat_.template block<1, vel_dim_>(i, placement_vel_dim_ + i * vel_dim_) =
        -1 * svm_grad.transpose() * rel_vel_mat_suc;
    qp_coeff_.ineq_vec_.template segment<1>(i) << calcSVMValue<SamplingSpaceType>(
        rel_sample, svm_mo_->param, svm_mo_, svm_coeff_vec_, svm_sv_mat_) - config_.svm_thre;
  }
  qp_coeff_.ineq_mat_.rightCols(svm_ineq_dim + collision_ineq_dim).diagonal().head(svm_ineq_dim).setConstant(-1);

  // ROS_INFO_STREAM("qp_coeff_.obj_mat_:\n" << qp_coeff_.obj_mat_);
  // ROS_INFO_STREAM("qp_coeff_.obj_vec_:\n" << qp_coeff_.obj_vec_.transpose());
  // ROS_INFO_STREAM("qp_coeff_.ineq_mat_:\n" << qp_coeff_.ineq_mat_);
  // ROS_INFO_STREAM("qp_coeff_.ineq_vec_:\n" << qp_coeff_.ineq_vec_.transpose());

  // Solve QP
  Eigen::VectorXd vel_all = qp_solver_->solve(qp_coeff_);
  if (qp_solver_->solve_failed_) {
    vel_all.setZero();
  }

  // Integrate
  integrateVelToSample<PlacementSamplingSpaceType>(
      current_placement_sample_,
      vel_all.template head<placement_vel_dim_>());
  for (int i = 0; i < config_.reaching_num; i++) {
    integrateVelToSample<SamplingSpaceType>(
        current_reaching_sample_list_[i],
        vel_all.template segment<vel_dim_>(placement_vel_dim_ + i * vel_dim_));
  }

  if (publish) {
    // Publish
    publishMarkerArray();
    publishCurrentState();
  }
}

template <SamplingSpace SamplingSpaceType>
void RmapPlanningPlacement<SamplingSpaceType>::runLoop(
    const std::shared_ptr<OmgCore::Robot>& rb)
{
  setup(rb);

  ros::Rate rate(config_.loop_rate);
  int loop_idx = 0;
  while (ros::ok()) {
    runOnce(loop_idx % config_.publish_interval == 0);

    rate.sleep();
    ros::spinOnce();
    loop_idx++;
  }
}

template <SamplingSpace SamplingSpaceType>
void RmapPlanningPlacement<SamplingSpaceType>::publishMarkerArray() const
{
  std_msgs::Header header_msg;
  header_msg.frame_id = "world";
  header_msg.stamp = ros::Time::now();

  // Instantiate marker array
  visualization_msgs::MarkerArray marker_arr_msg;

  // Delete marker
  visualization_msgs::Marker del_marker;
  del_marker.action = visualization_msgs::Marker::DELETEALL;
  del_marker.header = header_msg;
  del_marker.id = marker_arr_msg.markers.size();
  marker_arr_msg.markers.push_back(del_marker);

  // Reachable grids marker
  if (grid_set_msg_) {
    const GridPos<SamplingSpaceType>& grid_pos_min = getGridPosMin<SamplingSpaceType>(sample_min_);
    const GridPos<SamplingSpaceType>& grid_pos_range = getGridPosRange<SamplingSpaceType>(sample_min_, sample_max_);

    visualization_msgs::Marker grids_marker;
    grids_marker.header = header_msg;
    grids_marker.type = visualization_msgs::Marker::CUBE_LIST;
    grids_marker.scale = OmgCore::toVector3Msg(
        calcGridCubeScale<SamplingSpaceType>(grid_set_msg_->divide_nums, sample_max_ - sample_min_));
    grids_marker.color = OmgCore::toColorRGBAMsg({0.8, 0.0, 0.0, 0.3});

    for (int i = 0; i < config_.reaching_num; i++) {
      grids_marker.ns = "reachable_grids_" + std::to_string(i);
      grids_marker.id = marker_arr_msg.markers.size();
      grids_marker.pose = OmgCore::toPoseMsg(
          sampleToPose<PlacementSamplingSpaceType>(current_placement_sample_));
      const SampleType& slice_sample =
          relSample<SamplingSpaceType>(current_placement_sample_, current_reaching_sample_list_[i]);
      GridIdxs<SamplingSpaceType> slice_divide_idxs;
      gridDivideRatiosToIdxs(
          slice_divide_idxs,
          (sampleToGridPos<SamplingSpaceType>(slice_sample) - grid_pos_min).array() / grid_pos_range.array(),
          grid_set_msg_->divide_nums);
      std::vector<int> slice_update_dims(std::min(2, sample_dim_));
      std::iota(slice_update_dims.begin(), slice_update_dims.end(), 0);
      grids_marker.points.clear();
      loopGrid<SamplingSpaceType>(
          grid_set_msg_->divide_nums,
          grid_pos_min,
          grid_pos_range,
          [&](int grid_idx, const GridPos<SamplingSpaceType>& grid_pos) {
            if (grid_set_msg_->values[grid_idx] > config_.svm_thre) {
              Eigen::Vector3d pos = sampleToCloudPos<SamplingSpaceType>(gridPosToSample<SamplingSpaceType>(grid_pos));
              if constexpr (!(SamplingSpaceType == SamplingSpace::R3 ||
                              SamplingSpaceType == SamplingSpace::SE3)) {
                  pos.z() = 0;
                }
              grids_marker.points.push_back(OmgCore::toPointMsg(pos));
            }
          },
          slice_update_dims,
          slice_divide_idxs);
      marker_arr_msg.markers.push_back(grids_marker);
    }
  }

  marker_arr_pub_.publish(marker_arr_msg);
}

template <SamplingSpace SamplingSpaceType>
void RmapPlanningPlacement<SamplingSpaceType>::publishCurrentState() const
{
  std_msgs::Header header_msg;
  header_msg.frame_id = "world";
  header_msg.stamp = ros::Time::now();

  // Publish pose (placement pose)
  geometry_msgs::PoseStamped pose_msg;
  pose_msg.header = header_msg;
  pose_msg.pose = OmgCore::toPoseMsg(
      sampleToPose<PlacementSamplingSpaceType>(current_placement_sample_));
  current_pose_pub_.publish(pose_msg);

  // Publish pose array (reaching poses)
  geometry_msgs::PoseArray pose_arr_msg;
  pose_arr_msg.header = header_msg;
  pose_arr_msg.poses.resize(config_.reaching_num);
  for (int i = 0; i < config_.reaching_num; i++) {
    pose_arr_msg.poses[i] = OmgCore::toPoseMsg(
        sampleToPose<SamplingSpaceType>(current_reaching_sample_list_[i]));
  }
  current_pose_arr_pub_.publish(pose_arr_msg);
}

template <SamplingSpace SamplingSpaceType>
void RmapPlanningPlacement<SamplingSpaceType>::transCallback(
    const geometry_msgs::TransformStamped::ConstPtr& trans_st_msg)
{
  std::string frame_id = trans_st_msg->child_frame_id;
  if (frame_id == "target_placement") {
    target_placement_sample_ =
        poseToSample<PlacementSamplingSpaceType>(OmgCore::toSvaPTransform(trans_st_msg->transform));
  } else if (frame_id == "target") {
    sva::PTransformd center_pose = OmgCore::toSvaPTransform(trans_st_msg->transform);
    for (int i = 0; i < config_.reaching_num; i++) {
      double angle = M_PI * i / (config_.reaching_num - 1);
      sva::PTransformd target_pose(
          center_pose.rotation(),
          (sva::PTransformd(Eigen::Vector3d(config_.target_traj_radius, 0, 0)) *
           sva::PTransformd(sva::RotZ(angle)) * center_pose).translation());
      target_reaching_sample_list_[i] = poseToSample<SamplingSpaceType>(target_pose);
    }
  }
}

template <SamplingSpace SamplingSpaceType>
bool RmapPlanningPlacement<SamplingSpaceType>::postureCallback(
    std_srvs::Empty::Request& req,
    std_srvs::Empty::Response& res)
{
  if (!problem_) {
    ROS_ERROR_STREAM("[postureCallback] IK problem is not initialized.");
    return false;
  }

  const auto& rb = rb_arr_[0];
  const auto& rbc = problem_->rbcArr()[0];
  OmgCore::AuxRobotArray aux_rb_arr;
  rb->jvel_max_scale_ = 1e10;

  // Setup random sampling of joint position (used for initial posture)
  std::vector<int> joint_idx_list;
  for (const auto& joint : rb->joints()) {
    const auto& joint_name = joint.name();
    int joint_idx = rb->jointIndexByName(joint_name);

    if (joint_name == "Root" ||
        joint.dof() != 1 ||
        std::find(config_.ik_exclude_joint_name_list.begin(), config_.ik_exclude_joint_name_list.end(),
                  joint_name) != config_.ik_exclude_joint_name_list.end()) {
      // Overwrite joint range to restrict joints to be used
      // Be carefull that this overwrites original robot
      // This becomes unnecessary when optmotiongen supports joint selection
      std::fill(rb->jposs_min_[joint_idx].begin(), rb->jposs_min_[joint_idx].end(), 0);
      std::fill(rb->jposs_max_[joint_idx].begin(), rb->jposs_max_[joint_idx].end(), 0);
    } else {
      joint_idx_list.push_back(joint_idx);
    }
  }

  Eigen::VectorXd joint_pos_coeff;
  Eigen::VectorXd joint_pos_offset;
  joint_pos_coeff.resize(joint_idx_list.size());
  joint_pos_offset.resize(joint_idx_list.size());
  for (size_t i = 0; i < joint_idx_list.size(); i++) {
    const auto& joint_name = rb->joint(joint_idx_list[i]).name();
    double lower_joint_pos = rb->limits_.lower.at(joint_name)[0];
    double upper_joint_pos = rb->limits_.upper.at(joint_name)[0];
    joint_pos_coeff[i] = (upper_joint_pos - lower_joint_pos) / 2;
    joint_pos_offset[i] = (upper_joint_pos + lower_joint_pos) / 2;
  }

  // Solve IK
  optmotiongen_msgs::RobotStateArray robot_state_arr_msg;
  for (int i = 0; i < config_.reaching_num; i++) {
    // Set IK target
    rb->rootPose(sampleToPose<PlacementSamplingSpaceType>(current_placement_sample_));
    body_task_->target() = sampleToPose<SamplingSpaceType>(current_reaching_sample_list_[i]);

    bool ik_solved = false;
    double best_error = std::numeric_limits<double>::max();
    Eigen::VectorXd best_error_vec(0);
    std::shared_ptr<rbd::MultiBodyConfig> best_rbc;
    for (int j = 0; j < config_.ik_trial_num; j++) {
      if (j == 0) {
        // Set zero configuration
        rbc->zero(*rb);
      } else {
        // Set random configuration
        Eigen::VectorXd joint_pos =
            joint_pos_coeff.cwiseProduct(Eigen::VectorXd::Random(joint_idx_list.size())) + joint_pos_offset;
        for (size_t k = 0; k < joint_idx_list.size(); k++) {
          rbc->q[joint_idx_list[k]][0] = joint_pos[k];
        }
      }
      rbd::forwardKinematics(*rb, *rbc);

      // Solve IK
      problem_->run(config_.ik_loop_num);
      taskset_.update(rb_arr_, problem_->rbcArr(), aux_rb_arr);

      if (taskset_.errorSquaredNorm(false) < best_error) {
        best_error = taskset_.errorSquaredNorm(false);
        best_error_vec = body_task_->weight().cwiseProduct(body_task_->value());
        best_rbc = std::make_shared<rbd::MultiBodyConfig>(*rbc);
      }
      if (taskset_.errorSquaredNorm(false) < std::pow(config_.ik_error_thre, 2)) {
        ik_solved = true;
        break;
      }
    }

    if (!ik_solved) {
      ROS_WARN_STREAM("Failed to solve IK for reaching point " << std::to_string(i) << ". Task error: "
                      << std::sqrt(best_error) << " [" << best_error_vec.transpose() << "]");
    }

    // Add robot state message
    robot_state_arr_msg.robot_states.push_back(rb->makeRobotStateMsg(best_rbc));
  }

  // Publish robot
  rs_arr_pub_.publish(robot_state_arr_msg);

  return true;
}

template <SamplingSpace SamplingSpaceType>
bool RmapPlanningPlacement<SamplingSpaceType>::animateCallback(
    std_srvs::Empty::Request& req,
    std_srvs::Empty::Response& res)
{
  if (!problem_) {
    ROS_ERROR_STREAM("[animateCallback] IK problem is not initialized.");
    return false;
  }

  const auto& rb = rb_arr_[0];
  const auto& rbc = problem_->rbcArr()[0];
  OmgCore::AuxRobotArray aux_rb_arr;

  // Setup random sampling of joint position (used for initial posture)
  std::vector<int> joint_idx_list;
  for (const auto& joint : rb->joints()) {
    const auto& joint_name = joint.name();
    int joint_idx = rb->jointIndexByName(joint_name);

    if (joint_name == "Root" ||
        joint.dof() != 1 ||
        std::find(config_.ik_exclude_joint_name_list.begin(), config_.ik_exclude_joint_name_list.end(),
                  joint_name) != config_.ik_exclude_joint_name_list.end()) {
      // Overwrite joint range to restrict joints to be used
      // Be carefull that this overwrites original robot
      // This becomes unnecessary when optmotiongen supports joint selection
      std::fill(rb->jposs_min_[joint_idx].begin(), rb->jposs_min_[joint_idx].end(), 0);
      std::fill(rb->jposs_max_[joint_idx].begin(), rb->jposs_max_[joint_idx].end(), 0);
    } else {
      joint_idx_list.push_back(joint_idx);
    }
  }

  Eigen::VectorXd joint_pos_coeff;
  Eigen::VectorXd joint_pos_offset;
  joint_pos_coeff.resize(joint_idx_list.size());
  joint_pos_offset.resize(joint_idx_list.size());
  for (size_t i = 0; i < joint_idx_list.size(); i++) {
    const auto& joint_name = rb->joint(joint_idx_list[i]).name();
    double lower_joint_pos = rb->limits_.lower.at(joint_name)[0];
    double upper_joint_pos = rb->limits_.upper.at(joint_name)[0];
    joint_pos_coeff[i] = (upper_joint_pos - lower_joint_pos) / 2;
    joint_pos_offset[i] = (upper_joint_pos + lower_joint_pos) / 2;
  }

  // Set initial configuration
  rb->jvel_max_scale_ = 1e10;
  rb->rootPose(sampleToPose<PlacementSamplingSpaceType>(current_placement_sample_));
  body_task_->target() = sampleToPose<SamplingSpaceType>(current_reaching_sample_list_[0]);
  for (int i = 0; i < config_.ik_trial_num; i++) {
    if (i == 0) {
      // Set zero configuration
      rbc->zero(*rb);
    } else {
      // Set random configuration
      Eigen::VectorXd joint_pos =
          joint_pos_coeff.cwiseProduct(Eigen::VectorXd::Random(joint_idx_list.size())) + joint_pos_offset;
      for (size_t j = 0; j < joint_idx_list.size(); j++) {
        rbc->q[joint_idx_list[j]][0] = joint_pos[j];
      }
    }
    rbd::forwardKinematics(*rb, *rbc);

    // Solve IK
    problem_->run(config_.ik_loop_num);
    taskset_.update(rb_arr_, problem_->rbcArr(), aux_rb_arr);

    if (taskset_.errorSquaredNorm(false) < std::pow(config_.ik_error_thre, 2)) {
      break;
    } else if (i == config_.ik_trial_num - 1) {
      ROS_WARN("[animateCallback] Failed to generate initial posture.");
    }
  }

  // Loop for trajectory
  ros::Rate rate(config_.animate_adjacent_divide_num / config_.animate_adjacent_duration);
  rb->jvel_max_scale_ = config_.animate_adjacent_duration / config_.animate_adjacent_divide_num;
  for (int i = 0; i < config_.reaching_num - 1; i++) {
    const sva::PTransformd& pre_pose = sampleToPose<SamplingSpaceType>(current_reaching_sample_list_[i]);
    const sva::PTransformd& suc_pose = sampleToPose<SamplingSpaceType>(current_reaching_sample_list_[i + 1]);

    for (size_t j = 0; j < config_.animate_adjacent_divide_num; j++) {
      // Set IK target
      body_task_->target() = sva::interpolate(
          pre_pose, suc_pose, static_cast<double>(j + 1) / config_.animate_adjacent_divide_num);

      // Solve IK
      problem_->run(config_.animate_ik_loop_num);
      taskset_.update(rb_arr_, problem_->rbcArr(), aux_rb_arr);
      if (taskset_.errorSquaredNorm(false) > std::pow(config_.ik_error_thre, 2)) {
        ROS_WARN_STREAM_THROTTLE(
            10, "[animateCallback] IK error is large: " << std::sqrt(taskset_.errorSquaredNorm(false)));
      }

      // Publish robot
      rs_arr_pub_.publish(rb_arr_.makeRobotStateArrayMsg(problem_->rbcArr()));

      rate.sleep();
    }
  }

  return true;
}

std::shared_ptr<RmapPlanningBase> DiffRmap::createRmapPlanningPlacement(
    SamplingSpace sampling_space,
    const std::string& svm_path,
    const std::string& bag_path)
{
  if (sampling_space == SamplingSpace::R2) {
    return std::make_shared<RmapPlanningPlacement<SamplingSpace::R2>>(svm_path, bag_path);
  } else if (sampling_space == SamplingSpace::SO2) {
    return std::make_shared<RmapPlanningPlacement<SamplingSpace::SO2>>(svm_path, bag_path);
  } else if (sampling_space == SamplingSpace::SE2) {
    return std::make_shared<RmapPlanningPlacement<SamplingSpace::SE2>>(svm_path, bag_path);
  } else if (sampling_space == SamplingSpace::R3) {
    return std::make_shared<RmapPlanningPlacement<SamplingSpace::R3>>(svm_path, bag_path);
  } else if (sampling_space == SamplingSpace::SO3) {
    return std::make_shared<RmapPlanningPlacement<SamplingSpace::SO3>>(svm_path, bag_path);
  } else if (sampling_space == SamplingSpace::SE3) {
    return std::make_shared<RmapPlanningPlacement<SamplingSpace::SE3>>(svm_path, bag_path);
  } else {
    mc_rtc::log::error_and_throw<std::runtime_error>(
        "[createRmapPlanningPlacement] Unsupported SamplingSpace: {}", std::to_string(sampling_space));
  }
}

// Declare template specialized class
// See https://stackoverflow.com/a/8752879
template class RmapPlanningPlacement<SamplingSpace::R2>;
template class RmapPlanningPlacement<SamplingSpace::SO2>;
template class RmapPlanningPlacement<SamplingSpace::SE2>;
template class RmapPlanningPlacement<SamplingSpace::R3>;
template class RmapPlanningPlacement<SamplingSpace::SO3>;
template class RmapPlanningPlacement<SamplingSpace::SE3>;
