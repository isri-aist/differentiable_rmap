/* Author: Masaki Murooka */

#include <numeric>
#include <chrono>

#include <mc_rtc/constants.h>

#include <geometry_msgs/PoseArray.h>
#include <visualization_msgs/MarkerArray.h>

#include <optmotiongen/Utils/RosUtils.h>

#include <differentiable_rmap/RmapPlanningPlacement.h>
#include <differentiable_rmap/SVMUtils.h>
#include <differentiable_rmap/GridUtils.h>
#include <differentiable_rmap/libsvm_hotfix.h>

using namespace DiffRmap;


template <SamplingSpace SamplingSpaceType>
RmapPlanningPlacement<SamplingSpaceType>::RmapPlanningPlacement(
    const std::string& svm_path,
    const std::string& bag_path):
    RmapPlanning<SamplingSpaceType>(svm_path, bag_path)
{
  current_pose_arr_pub_ = nh_.template advertise<geometry_msgs::PoseArray>(
      "current_pose_arr", 1, true);
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
void RmapPlanningPlacement<SamplingSpaceType>::setup()
{
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
    SampleType sample_range = sample_max_ - sample_min_;
    visualization_msgs::Marker grids_marker;
    grids_marker.header = header_msg;
    grids_marker.type = visualization_msgs::Marker::CUBE_LIST;
    grids_marker.scale = OmgCore::toVector3Msg(
        calcGridCubeScale<SamplingSpaceType>(grid_set_msg_->divide_nums, sample_range));
    grids_marker.color = OmgCore::toColorRGBAMsg({0.8, 0.0, 0.0, 0.3});

    for (int i = 0; i < config_.reaching_num; i++) {
      grids_marker.ns = "reachable_grids_" + std::to_string(i);
      grids_marker.id = marker_arr_msg.markers.size();
      grids_marker.pose = OmgCore::toPoseMsg(
          sampleToPose<PlacementSamplingSpaceType>(current_placement_sample_));
      const SampleType& slice_sample =
          relSample<SamplingSpaceType>(current_placement_sample_, current_reaching_sample_list_[i]);
      GridIdxsType<SamplingSpaceType> slice_divide_idxs;
      gridDivideRatiosToIdxs(
          slice_divide_idxs,
          (slice_sample - sample_min_).array() / sample_range.array(),
          grid_set_msg_->divide_nums);
      std::vector<int> slice_update_dims(std::min(2, sample_dim_));
      std::iota(slice_update_dims.begin(), slice_update_dims.end(), 0);
      grids_marker.points.clear();
      loopGrid<SamplingSpaceType>(
          grid_set_msg_->divide_nums,
          sample_min_,
          sample_range,
          [&](int grid_idx, const SampleType& sample) {
            if (grid_set_msg_->values[grid_idx] > config_.svm_thre) {
              Eigen::Vector3d pos = sampleToCloudPos<SamplingSpaceType>(sample);
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
  } else if (frame_id.find("target_") == 0) {
    frame_id.erase(0, std::string("target_").length());
    target_reaching_sample_list_[std::stoi(frame_id)] =
        poseToSample<SamplingSpaceType>(OmgCore::toSvaPTransform(trans_st_msg->transform));
  }
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
