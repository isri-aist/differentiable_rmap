/* Author: Masaki Murooka */

#include <array>
#include <numeric>
#include <chrono>

#include <mc_rtc/constants.h>

#include <geometry_msgs/PoseArray.h>
#include <visualization_msgs/MarkerArray.h>
#include <jsk_recognition_msgs/PolygonArray.h>

#include <optmotiongen/Utils/RosUtils.h>
#include <optmotiongen/Func/CollisionFunc.h>

#include <differentiable_rmap/RmapPlanningFootstep.h>
#include <differentiable_rmap/SVMUtils.h>
#include <differentiable_rmap/GridUtils.h>
#include <differentiable_rmap/libsvm_hotfix.h>

using namespace DiffRmap;


namespace
{
/** \brief Calculate Jacobian about the position w.r.t. sample.
    \tparam SamplingSpaceType sampling space
    \param sample sample
    \param pos focused position represented in world frame
*/
template <SamplingSpace SamplingSpaceType>
Eigen::Matrix<double, 3, velDim<SamplingSpaceType>()> posJacobian(
    const Sample<SamplingSpaceType>& sample,
    const Eigen::Vector3d& pos)
{
  mc_rtc::log::error_and_throw<std::runtime_error>(
      "[posJacobian] Need to specialize for SamplingSpace {}", std::to_string(SamplingSpaceType));
  return Eigen::Matrix<double, 3, velDim<SamplingSpaceType>()>::Zero();
}

template <>
inline Eigen::Matrix<double, 3, velDim<SamplingSpace::SE2>()> posJacobian<SamplingSpace::SE2>(
    const Sample<SamplingSpace::SE2>& sample,
    const Eigen::Vector3d& pos)
{
  Eigen::Matrix<double, 3, velDim<SamplingSpace::SE2>()> jac
      = Eigen::Matrix<double, 3, velDim<SamplingSpace::SE2>()>::Zero();
  jac(0, 0) = 1;
  jac(1, 1) = 1;
  Eigen::Vector3d delta_pos = Eigen::Vector3d::Zero();
  delta_pos.template head<2>() = (pos - sample).template head<2>();
  jac.col(2) = Eigen::Vector3d::UnitZ().cross(delta_pos);
  return jac;
}
}

template <SamplingSpace SamplingSpaceType>
RmapPlanningFootstep<SamplingSpaceType>::RmapPlanningFootstep(
    const std::string& svm_path,
    const std::string& bag_path):
    RmapPlanning<SamplingSpaceType>(svm_path, bag_path)
{
  current_pose_arr_pub_ = nh_.template advertise<geometry_msgs::PoseArray>(
      "current_pose_arr", 1, true);
  current_poly_arr_pub_ = nh_.template advertise<jsk_recognition_msgs::PolygonArray>(
      "current_poly_arr", 1, true);
  if constexpr (isAlternateSupported()) {
      current_left_poly_arr_pub_ = nh_.template advertise<jsk_recognition_msgs::PolygonArray>(
          "current_left_poly_arr", 1, true);
      current_right_poly_arr_pub_ = nh_.template advertise<jsk_recognition_msgs::PolygonArray>(
          "current_right_poly_arr", 1, true);
    }
}

template <SamplingSpace SamplingSpaceType>
RmapPlanningFootstep<SamplingSpaceType>::~RmapPlanningFootstep()
{
  if (svm_mo_) {
    delete svm_mo_;
  }
}

template <SamplingSpace SamplingSpaceType>
void RmapPlanningFootstep<SamplingSpaceType>::configure(
    const mc_rtc::Configuration& mc_rtc_config)
{
  RmapPlanning<SamplingSpaceType>::configure(mc_rtc_config);

  config_.load(mc_rtc_config);
}

template <SamplingSpace SamplingSpaceType>
void RmapPlanningFootstep<SamplingSpaceType>::setup()
{
  // Setup QP coefficients and solver
  int config_dim = config_.footstep_num * vel_dim_;
  int svm_ineq_dim = config_.footstep_num;
  int collision_ineq_dim = config_.obst_shape_config_list.size() * config_.footstep_num;
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

  // Setup current sample sequence
  current_sample_seq_.resize(config_.footstep_num);
  sva::PTransformd accum_initial_sample_pose = sva::PTransformd::Identity();
  for (int i = 0; i < config_.footstep_num; i++) {
    if constexpr (isAlternateSupported()) {
        sva::PTransformd initial_sample_pose = config_.initial_sample_pose;
        if (config_.alternate_lr && (i % 2 == 1)) {
          initial_sample_pose.translation().y() *= -1;
          Eigen::Vector3d euler_angles = initial_sample_pose.rotation().transpose().eulerAngles(2, 1, 0);
          initial_sample_pose.rotation() = (Eigen::AngleAxisd(-1 * euler_angles[0], Eigen::Vector3d::UnitZ())
                                            * Eigen::AngleAxisd(euler_angles[1], Eigen::Vector3d::UnitY())
                                            * Eigen::AngleAxisd(euler_angles[2], Eigen::Vector3d::UnitX())
                                            ).toRotationMatrix().transpose();
        }
        accum_initial_sample_pose = initial_sample_pose * accum_initial_sample_pose;
      } else {
      accum_initial_sample_pose = config_.initial_sample_pose * accum_initial_sample_pose;
    }
    current_sample_seq_[i] = poseToSample<SamplingSpaceType>(accum_initial_sample_pose);
  }

  // Setup adjacent regularization
  adjacent_reg_mat_.setZero(config_dim, config_dim);
  for (int i = 0; i < config_.footstep_num; i++) {
    adjacent_reg_mat_.block<vel_dim_, vel_dim_>(i * vel_dim_, i * vel_dim_).diagonal().setConstant(
        (i == config_.footstep_num - 1 ? 1 : 2) * config_.adjacent_reg_weight);
    if (i != config_.footstep_num - 1) {
      adjacent_reg_mat_.block<vel_dim_, vel_dim_>((i + 1) * vel_dim_, i * vel_dim_).diagonal().setConstant(
          -config_.adjacent_reg_weight);
      adjacent_reg_mat_.block<vel_dim_, vel_dim_>(i * vel_dim_, (i + 1) * vel_dim_).diagonal().setConstant(
          -config_.adjacent_reg_weight);
    }
  }
  // ROS_INFO_STREAM("adjacent_reg_mat_:\n" << adjacent_reg_mat_);

  // Setup collision
  foot_sch_ = std::make_shared<sch::S_Box>(
      config_.foot_shape_config.scale.x(), config_.foot_shape_config.scale.y(), config_.foot_shape_config.scale.z());
  const int& obst_num = config_.obst_shape_config_list.size();
  obst_sch_list_.resize(obst_num);
  sch_cd_list_.resize(obst_num);
  closest_points_list_.resize(obst_num * config_.footstep_num);
  for (size_t i = 0; i < obst_num; i++) {
    const auto& obst_shape_config = config_.obst_shape_config_list[i];
    obst_sch_list_[i] = std::make_shared<sch::S_Box>(
        obst_shape_config.scale.x(), obst_shape_config.scale.y(), obst_shape_config.scale.z());
    OmgCore::setSchObjPose(obst_sch_list_[i], obst_shape_config.pose);
    sch_cd_list_[i] = std::make_shared<sch::CD_Pair>(foot_sch_.get(), obst_sch_list_[i].get());
  }
}

template <SamplingSpace SamplingSpaceType>
void RmapPlanningFootstep<SamplingSpaceType>::runOnce(bool publish)
{
  int config_dim = config_.footstep_num * vel_dim_;
  int svm_ineq_dim = config_.footstep_num;
  int collision_ineq_dim = config_.obst_shape_config_list.size() * config_.footstep_num;

  // Set QP objective matrices
  qp_coeff_.obj_mat_.setZero();
  qp_coeff_.obj_vec_.setZero();
  const VelType& sample_error = sampleError<SamplingSpaceType>(target_sample_, current_sample_seq_.back());
  qp_coeff_.obj_mat_.diagonal().template segment<vel_dim_>(config_dim - vel_dim_).setConstant(1.0);
  qp_coeff_.obj_mat_.diagonal().head(config_dim).array() += sample_error.squaredNorm() + config_.reg_weight;
  qp_coeff_.obj_mat_.diagonal().tail(svm_ineq_dim + collision_ineq_dim).head(
      svm_ineq_dim).setConstant(config_.svm_ineq_weight);
  qp_coeff_.obj_mat_.diagonal().tail(svm_ineq_dim + collision_ineq_dim).tail(
      collision_ineq_dim).setConstant(config_.collision_ineq_weight);
  qp_coeff_.obj_vec_.template segment<vel_dim_>(config_dim - vel_dim_) = sample_error;
  Eigen::VectorXd current_config(config_dim);
  for (int i = 0; i < config_.footstep_num; i++) {
    // The implementation of adjacent regularization is not strict because the error between samples is not a simple subtraction
    current_config.template segment<vel_dim_>(i * vel_dim_) =
        sampleError<SamplingSpaceType>(identity_sample_, current_sample_seq_[i]);
  }
  qp_coeff_.obj_vec_.head(config_dim) += adjacent_reg_mat_ * current_config;
  qp_coeff_.obj_mat_.topLeftCorner(config_dim, config_dim) += adjacent_reg_mat_;

  // Set QP inequality matrices of reachability
  qp_coeff_.ineq_mat_.setZero();
  qp_coeff_.ineq_vec_.setZero();
  for (int i = 0; i < config_.footstep_num; i++) {
    const SampleType& pre_sample =
        i == 0 ? poseToSample<SamplingSpaceType>(sva::PTransformd::Identity()) : current_sample_seq_[i - 1];
    const SampleType& suc_sample = current_sample_seq_[i];
    SampleType rel_sample = relSample<SamplingSpaceType>(pre_sample, suc_sample);
    if constexpr (isAlternateSupported()) {
        if (config_.alternate_lr && (i % 2 == 1)) {
          rel_sample.template tail<2>() *= -1;
        }
      }
    const VelType& svm_grad = calcSVMGrad<SamplingSpaceType>(
            rel_sample, svm_mo_->param, svm_mo_, svm_coeff_vec_, svm_sv_mat_);
    VelToVelMat<SamplingSpaceType> rel_vel_mat_suc = relVelToVelMat<SamplingSpaceType>(pre_sample, suc_sample, true);
    if constexpr (isAlternateSupported()) {
        if (config_.alternate_lr && (i % 2 == 1)) {
          rel_vel_mat_suc.template bottomRows<2>() *= -1;
        }
      }
    qp_coeff_.ineq_mat_.template block<1, vel_dim_>(i, i * vel_dim_) = -1 * svm_grad.transpose() * rel_vel_mat_suc;
    qp_coeff_.ineq_vec_.template segment<1>(i) << calcSVMValue<SamplingSpaceType>(
        rel_sample, svm_mo_->param, svm_mo_, svm_coeff_vec_, svm_sv_mat_) - config_.svm_thre;
    if (i > 0) {
      VelToVelMat<SamplingSpaceType> rel_vel_mat_pre = relVelToVelMat<SamplingSpaceType>(pre_sample, suc_sample, false);
      if constexpr (isAlternateSupported()) {
          if (config_.alternate_lr && (i % 2 == 1)) {
            rel_vel_mat_pre.template bottomRows<2>() *= -1;
          }
        }
      qp_coeff_.ineq_mat_.template block<1, vel_dim_>(i, (i - 1) * vel_dim_) = -1 * svm_grad.transpose() * rel_vel_mat_pre;
    }
  }
  qp_coeff_.ineq_mat_.rightCols(svm_ineq_dim + collision_ineq_dim).diagonal().head(svm_ineq_dim).setConstant(-1);

  // Set QP inequality matrices of collision
  std::array<sch::Point3, 2> closest_sch_points;
  for (int i = 0; i < config_.footstep_num; i++) {
    OmgCore::setSchObjPose(
        foot_sch_, config_.foot_shape_config.pose * sampleToPose<SamplingSpaceType>(current_sample_seq_[i]));
    for (size_t j = 0; j < config_.obst_shape_config_list.size(); j++) {
      int idx = i * config_.obst_shape_config_list.size() + j;
      double signed_dist = sch_cd_list_[j]->getClosestPoints(closest_sch_points[0], closest_sch_points[1]);
      // getClosestPoints() returns the squared distance with sign
      signed_dist = signed_dist >= 0 ? std::sqrt(signed_dist) : -std::sqrt(-signed_dist);
      std::array<Eigen::Vector3d, 2>& closest_points = closest_points_list_[idx];
      for (auto k : {0, 1}) {
        closest_points[k] << closest_sch_points[k][0], closest_sch_points[k][1], closest_sch_points[k][2];
      }
      // Skip updating collision_dir_ when signed_dist is zero
      if (std::abs(signed_dist) > 1e-10) {
        collision_dir_ = (closest_points[0] - closest_points[1]) / signed_dist;
      }
      // If collision_dir_ is zero vector (initial value), skip the corresponding inequality constraint
      if (collision_dir_.norm() == 0.0) {
        continue;
      }
      qp_coeff_.ineq_mat_.template block<1, vel_dim_>(config_.footstep_num + idx, i * vel_dim_) =
          -1 * collision_dir_.transpose() * posJacobian<SamplingSpaceType>(current_sample_seq_[i], closest_points[0]);
      qp_coeff_.ineq_vec_.template segment<1>(config_.footstep_num + idx) << signed_dist - config_.collision_margin;
    }
  }
  qp_coeff_.ineq_mat_.rightCols(svm_ineq_dim + collision_ineq_dim).diagonal().tail(collision_ineq_dim).setConstant(-1);

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
  for (int i = 0; i < config_.footstep_num; i++) {
    integrateVelToSample<SamplingSpaceType>(
        current_sample_seq_[i], vel_all.template segment<vel_dim_>(i * vel_dim_));
  }

  if (publish) {
    // Publish
    publishMarkerArray();
    publishCurrentState();
  }
}

template <SamplingSpace SamplingSpaceType>
void RmapPlanningFootstep<SamplingSpaceType>::publishMarkerArray() const
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
    grids_marker.scale.z = 0.01;
    grids_marker.color = OmgCore::toColorRGBAMsg({0.8, 0.0, 0.0, 0.3});

    for (int i = 0; i < config_.footstep_num; i++) {
      grids_marker.ns = "reachable_grids_" + std::to_string(i);
      grids_marker.id = marker_arr_msg.markers.size();
      grids_marker.pose = OmgCore::toPoseMsg(
          i == 0 ? sva::PTransformd::Identity() : sampleToPose<SamplingSpaceType>(current_sample_seq_[i - 1]));
      if constexpr (isAlternateSupported()) {
          grids_marker.color = OmgCore::toColorRGBAMsg(
              (config_.alternate_lr && (i % 2 == 1)) ?
              std::array<double, 4>{0.0, 0.8, 0.0, 0.3} : std::array<double, 4>{0.8, 0.0, 0.0, 0.3});
        }
      SampleType slice_sample =
          i == 0 ? current_sample_seq_[i] : relSample<SamplingSpaceType>(current_sample_seq_[i - 1], current_sample_seq_[i]);
      if constexpr (isAlternateSupported()) {
          if (config_.alternate_lr && (i % 2 == 1)) {
            slice_sample.template tail<2>() *= -1;
          }
        }
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
              pos.z() = 0;
              if constexpr (isAlternateSupported()) {
                  if (config_.alternate_lr && (i % 2 == 1)) {
                    pos.y() *= -1;
                  }
                }
              grids_marker.points.push_back(OmgCore::toPointMsg(pos));
            }
          },
          slice_update_dims,
          slice_divide_idxs);
      marker_arr_msg.markers.push_back(grids_marker);
    }
  }

  // Obstacle marker
  for (size_t i = 0; i < config_.obst_shape_config_list.size(); i++) {
    const auto& obst_shape_config = config_.obst_shape_config_list[i];
    visualization_msgs::Marker obst_marker;
    obst_marker.header = header_msg;
    obst_marker.ns = "obstacle_" + std::to_string(i);
    obst_marker.id = marker_arr_msg.markers.size();
    obst_marker.type = visualization_msgs::Marker::CUBE;
    obst_marker.pose = OmgCore::toPoseMsg(obst_shape_config.pose);
    obst_marker.scale = OmgCore::toVector3Msg(obst_shape_config.scale);
    obst_marker.scale.z = 0.005;
    obst_marker.color = OmgCore::toColorRGBAMsg({0.0, 0.0, 0.8, 0.5});
    marker_arr_msg.markers.push_back(obst_marker);
  }

  // Collision marker (connecting the closest points)
  visualization_msgs::Marker collision_points_marker;
  collision_points_marker.header.frame_id = "world";
  collision_points_marker.ns = "collision_points";
  collision_points_marker.id = marker_arr_msg.markers.size();
  collision_points_marker.type = visualization_msgs::Marker::SPHERE_LIST;
  collision_points_marker.color = OmgCore::toColorRGBAMsg({0, 0, 1, 1});
  collision_points_marker.scale = OmgCore::toVector3Msg({0.02, 0.02, 0.02}); // sphere size
  collision_points_marker.pose.orientation = OmgCore::toQuaternionMsg({0, 0, 0, 1});
  visualization_msgs::Marker collision_lines_marker;
  collision_lines_marker.header.frame_id = "world";
  collision_lines_marker.ns = "collision_lines";
  collision_lines_marker.id = marker_arr_msg.markers.size();
  collision_lines_marker.type = visualization_msgs::Marker::LINE_LIST;
  collision_lines_marker.color = OmgCore::toColorRGBAMsg({0, 0, 1, 1});
  collision_lines_marker.scale.x = 0.01; // line width
  collision_lines_marker.pose.orientation = OmgCore::toQuaternionMsg({0, 0, 0, 1});
  for (int i = 0; i < config_.footstep_num; i++) {
    for (size_t j = 0; j < config_.obst_shape_config_list.size(); j++) {
      int idx = i * config_.obst_shape_config_list.size() + j;
      for (auto k : {0, 1}) {
        const auto& point_msg = OmgCore::toPointMsg(closest_points_list_[idx][k]);
        collision_points_marker.points.push_back(point_msg);
        collision_lines_marker.points.push_back(point_msg);
      }
    }
  }
  marker_arr_msg.markers.push_back(collision_points_marker);
  marker_arr_msg.markers.push_back(collision_lines_marker);

  marker_arr_pub_.publish(marker_arr_msg);
}

template <SamplingSpace SamplingSpaceType>
void RmapPlanningFootstep<SamplingSpaceType>::publishCurrentState() const
{
  std_msgs::Header header_msg;
  header_msg.frame_id = "world";
  header_msg.stamp = ros::Time::now();

  // Publish pose array
  geometry_msgs::PoseArray pose_arr_msg;
  pose_arr_msg.header = header_msg;
  pose_arr_msg.poses.resize(config_.footstep_num + 1);
  for (int i = 0; i < config_.footstep_num + 1; i++) {
    pose_arr_msg.poses[i] = OmgCore::toPoseMsg(
        i == 0 ? sva::PTransformd::Identity() : sampleToPose<SamplingSpaceType>(current_sample_seq_[i - 1]));
  }
  current_pose_arr_pub_.publish(pose_arr_msg);

  // Publish polygon array
  jsk_recognition_msgs::PolygonArray poly_arr_msg;
  jsk_recognition_msgs::PolygonArray left_poly_arr_msg;
  jsk_recognition_msgs::PolygonArray right_poly_arr_msg;
  poly_arr_msg.header = header_msg;
  poly_arr_msg.polygons.resize(config_.footstep_num + 1);
  if constexpr (isAlternateSupported()) {
      left_poly_arr_msg.header = header_msg;
      right_poly_arr_msg.header = header_msg;
    }
  for (int i = 0; i < config_.footstep_num + 1; i++) {
    poly_arr_msg.polygons[i].header = header_msg;
    sva::PTransformd foot_pose =
        i == 0 ? sva::PTransformd::Identity() : sampleToPose<SamplingSpaceType>(current_sample_seq_[i - 1]);
    poly_arr_msg.polygons[i].polygon.points.resize(config_.foot_vertices.size());
    for (size_t j = 0; j < config_.foot_vertices.size(); j++) {
      poly_arr_msg.polygons[i].polygon.points[j] =
          OmgCore::toPoint32Msg(foot_pose.rotation().transpose() * config_.foot_vertices[j] + foot_pose.translation());
    }
    if constexpr (isAlternateSupported()) {
        if (config_.alternate_lr) {
          if (i % 2 == 1) {
            left_poly_arr_msg.polygons.push_back(poly_arr_msg.polygons[i]);
          } else {
            right_poly_arr_msg.polygons.push_back(poly_arr_msg.polygons[i]);
          }
        }
      }
  }
  current_poly_arr_pub_.publish(poly_arr_msg);
  if constexpr (isAlternateSupported()) {
      if (config_.alternate_lr) {
        current_left_poly_arr_pub_.publish(left_poly_arr_msg);
        current_right_poly_arr_pub_.publish(right_poly_arr_msg);
      }
    }
}

std::shared_ptr<RmapPlanningBase> DiffRmap::createRmapPlanningFootstep(
    SamplingSpace sampling_space,
    const std::string& svm_path,
    const std::string& bag_path)
{
  if (sampling_space == SamplingSpace::R2) {
    return std::make_shared<RmapPlanningFootstep<SamplingSpace::R2>>(svm_path, bag_path);
  } else if (sampling_space == SamplingSpace::SO2) {
    return std::make_shared<RmapPlanningFootstep<SamplingSpace::SO2>>(svm_path, bag_path);
  } else if (sampling_space == SamplingSpace::SE2) {
    return std::make_shared<RmapPlanningFootstep<SamplingSpace::SE2>>(svm_path, bag_path);
  } else if (sampling_space == SamplingSpace::R3) {
    return std::make_shared<RmapPlanningFootstep<SamplingSpace::R3>>(svm_path, bag_path);
  } else if (sampling_space == SamplingSpace::SO3) {
    return std::make_shared<RmapPlanningFootstep<SamplingSpace::SO3>>(svm_path, bag_path);
  } else if (sampling_space == SamplingSpace::SE3) {
    return std::make_shared<RmapPlanningFootstep<SamplingSpace::SE3>>(svm_path, bag_path);
  } else {
    mc_rtc::log::error_and_throw<std::runtime_error>(
        "[createRmapPlanningFootstep] Unsupported SamplingSpace: {}", std::to_string(sampling_space));
  }
}

// Declare template specialized class
// See https://stackoverflow.com/a/8752879
template class RmapPlanningFootstep<SamplingSpace::R2>;
template class RmapPlanningFootstep<SamplingSpace::SO2>;
template class RmapPlanningFootstep<SamplingSpace::SE2>;
template class RmapPlanningFootstep<SamplingSpace::R3>;
template class RmapPlanningFootstep<SamplingSpace::SO3>;
template class RmapPlanningFootstep<SamplingSpace::SE3>;
