/* Author: Masaki Murooka */

#include <chrono>

#include <mc_rtc/constants.h>

#include <geometry_msgs/PoseArray.h>
#include <visualization_msgs/MarkerArray.h>
#include <jsk_recognition_msgs/PolygonArray.h>

#include <optmotiongen/Utils/RosUtils.h>

#include <differentiable_rmap/RmapPlanningMulticontact.h>
#include <differentiable_rmap/SVMUtils.h>
#include <differentiable_rmap/GridUtils.h>
#include <differentiable_rmap/libsvm_hotfix.h>

using namespace DiffRmap;


namespace
{
/** \brief Calculate SVM value.
    \tparam SamplingSpaceType sampling space
    \param sample sample
    \param rmap_planning shared pointer of rmap_planning
*/
template <SamplingSpace SamplingSpaceType>
double calcSVMValueWithRmapPlanning(
    const Sample<SamplingSpaceType>& sample,
    const std::shared_ptr<RmapPlanning<SamplingSpaceType>>& rmap_planning)
{
  return calcSVMValue<SamplingSpaceType>(
      sample,
      rmap_planning->svm_mo_->param,
      rmap_planning->svm_mo_,
      rmap_planning->svm_coeff_vec_,
      rmap_planning->svm_sv_mat_);
}

/** \brief Calculate gradient of SVM value.
    \tparam SamplingSpaceType sampling space
    \param sample sample
    \param rmap_planning shared pointer of rmap_planning
*/
template <SamplingSpace SamplingSpaceType>
Vel<SamplingSpaceType> calcSVMGradWithRmapPlanning(
    const Sample<SamplingSpaceType>& sample,
    const std::shared_ptr<RmapPlanning<SamplingSpaceType>>& rmap_planning)
{
  return calcSVMGrad<SamplingSpaceType>(
      sample,
      rmap_planning->svm_mo_->param,
      rmap_planning->svm_mo_,
      rmap_planning->svm_coeff_vec_,
      rmap_planning->svm_sv_mat_);
}

/** \brief Get relative sample from foot to hand which is represented in foot frame.
    \param foot_sample foot sample
    \param hand_sample hand sample
    \param nominal_z nominal z position
*/
Sample<SamplingSpace::R3> relSampleHandFromFoot(
    const Sample<SamplingSpace::SE2>& foot_sample,
    const Sample<SamplingSpace::R3>& hand_sample,
    double nominal_z)
{
  double cos = std::cos(foot_sample.z());
  double sin = std::sin(foot_sample.z());
  double dx = hand_sample.x() - foot_sample.x();
  double dy = hand_sample.y() - foot_sample.y();

  Sample<SamplingSpace::R3> rel_sample;
  rel_sample <<
      cos * dx + sin * dy,
      -sin * dx + cos * dy,
      hand_sample.z() - nominal_z;

  return rel_sample;
}

/** \brief Get gradient of relative sample from foot to hand which is represented in foot frame.
    \param foot_sample foot sample
    \param hand_sample hand sample
    \param wrt_hand if true, the returned matrix is w.r.t. the hand. otherwise, it is w.r.t. foot.
*/
Eigen::Matrix<double, sampleDim<SamplingSpace::R3>(), sampleDim<SamplingSpace::SE2>()>
relSampleGradHandFromFoot(
    const Sample<SamplingSpace::SE2>& foot_sample,
    const Sample<SamplingSpace::R3>& hand_sample,
    bool wrt_hand)
{
  double cos = std::cos(foot_sample.z());
  double sin = std::sin(foot_sample.z());

  Eigen::Matrix<double, sampleDim<SamplingSpace::R3>(), sampleDim<SamplingSpace::SE2>()> mat;
  mat <<
      cos, sin, 0,
      -sin, cos, 0,
      0, 0, 0;

  if (wrt_hand) {
    mat(2, 2) = 1;
  } else {
    double dx = hand_sample.x() - foot_sample.x();
    double dy = hand_sample.y() - foot_sample.y();
    mat *= -1;
    mat(0, 2) = -sin * dx + cos * dy;
    mat(1, 2) = -cos * dx - sin * dy;
  }

  return mat;
}
}

Limb DiffRmap::strToLimb(const std::string& limb_str)
{
  if (limb_str == "LeftFoot") {
    return Limb::LeftFoot;
  } else if (limb_str == "RightFoot") {
    return Limb::RightFoot;
  } else if (limb_str == "LeftHand") {
    return Limb::LeftHand;
  } else if (limb_str == "RightHand") {
    return Limb::RightHand;
  } else {
    mc_rtc::log::error_and_throw<std::runtime_error>(
        "[strToLimb] Unsupported Limb name: {}", limb_str);
  }
}

RmapPlanningMulticontact::RmapPlanningMulticontact(
    const std::unordered_map<Limb, std::string>& svm_path_list,
    const std::unordered_map<Limb, std::string>& bag_path_list)
{
  // Setup ROS
  trans_sub_ = nh_.subscribe(
      "interactive_marker_transform",
      100,
      &RmapPlanningMulticontact::transCallback,
      this);
  marker_arr_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("marker_arr", 1, true);
  current_pose_arr_pub_ = nh_.template advertise<geometry_msgs::PoseArray>(
      "current_pose_arr", 1, true);
  current_poly_arr_pub_ = nh_.template advertise<jsk_recognition_msgs::PolygonArray>(
      "current_poly_arr", 1, true);

  rmap_planning_list_[Limb::LeftFoot] = std::make_shared<RmapPlanning<FootSamplingSpaceType>>(
      svm_path_list.at(Limb::LeftFoot),
      bag_path_list.at(Limb::LeftFoot),
      false);
  rmap_planning_list_[Limb::RightFoot] = std::make_shared<RmapPlanning<FootSamplingSpaceType>>(
      svm_path_list.at(Limb::RightFoot),
      bag_path_list.at(Limb::RightFoot),
      false);
  rmap_planning_list_[Limb::LeftHand] = std::make_shared<RmapPlanning<HandSamplingSpaceType>>(
      svm_path_list.at(Limb::LeftHand),
      bag_path_list.at(Limb::LeftHand),
      false);
}

RmapPlanningMulticontact::~RmapPlanningMulticontact()
{
}

void RmapPlanningMulticontact::configure(const mc_rtc::Configuration& mc_rtc_config)
{
  mc_rtc_config_ = mc_rtc_config;
  config_.load(mc_rtc_config);
}

void RmapPlanningMulticontact::setup()
{
  // Setup QP coefficients and solver
  int config_dim = (config_.motion_len + 1) * foot_vel_dim_ + config_.motion_len * hand_vel_dim_;
  int svm_ineq_dim = 3 + 7 * (config_.motion_len - 1);
  int collision_ineq_dim = 0;
  int hand_start_idx = (config_.motion_len + 1) * foot_vel_dim_;
  // Introduce variables for inequality constraint errors
  qp_coeff_.setup(
      config_dim + svm_ineq_dim + collision_ineq_dim,
      config_.motion_len,
      svm_ineq_dim + collision_ineq_dim);
  qp_coeff_.x_min_.head(config_dim).setConstant(-config_.delta_config_limit);
  qp_coeff_.x_max_.head(config_dim).setConstant(config_.delta_config_limit);
  qp_coeff_.x_min_.tail(svm_ineq_dim + collision_ineq_dim).setConstant(-1e10);
  qp_coeff_.x_max_.tail(svm_ineq_dim + collision_ineq_dim).setConstant(1e10);

  qp_solver_ = OmgCore::allocateQpSolver(OmgCore::QpSolverType::JRLQP);

  // Setup current sample sequence
  current_foot_sample_seq_.resize(config_.motion_len + 1);
  current_hand_sample_seq_.resize(config_.motion_len);
  for (int i = 0; i < config_.motion_len + 1; i++) {
    current_foot_sample_seq_[i] = poseToSample<FootSamplingSpaceType>(
        config_.initial_sample_pose_list.at(i % 2 == 0 ? Limb::LeftFoot : Limb::RightFoot));
    if (i < config_.motion_len) {
      current_hand_sample_seq_[i] = poseToSample<HandSamplingSpaceType>(
          config_.initial_sample_pose_list.at(Limb::LeftHand));
    }
  }

  // Setup adjacent regularization
  adjacent_reg_mat_.setZero(config_dim, config_dim);
  for (int i = 0; i < config_.motion_len + 1; i++) {
    // Set for adjacent foot
    adjacent_reg_mat_.block<foot_vel_dim_, foot_vel_dim_>(i * foot_vel_dim_, i * foot_vel_dim_).diagonal().setConstant(
            ((i == 0 || i == config_.motion_len) ? 1 : 2) * config_.adjacent_reg_weight);
    if (i < config_.motion_len) {
      adjacent_reg_mat_.block<foot_vel_dim_, foot_vel_dim_>((i + 1) * foot_vel_dim_, i * foot_vel_dim_).diagonal().setConstant(
              -config_.adjacent_reg_weight);
      adjacent_reg_mat_.block<foot_vel_dim_, foot_vel_dim_>(i * foot_vel_dim_, (i + 1) * foot_vel_dim_).diagonal().setConstant(
              -config_.adjacent_reg_weight);
    }

    // Set for adjacent hand
    // if (i < config_.motion_len) {
    //   adjacent_reg_mat_.block<hand_vel_dim_, hand_vel_dim_>(
    //       hand_start_idx + i * hand_vel_dim_, hand_start_idx + i * hand_vel_dim_).diagonal().setConstant(
    //           ((i == 0 || i == config_.motion_len - 1) ? 1 : 2) * config_.adjacent_reg_weight);
    // }
    // if (i < config_.motion_len - 1) {
    //   adjacent_reg_mat_.block<hand_vel_dim_, hand_vel_dim_>(
    //       hand_start_idx + (i + 1) * hand_vel_dim_, hand_start_idx + i * hand_vel_dim_).diagonal().setConstant(
    //           -config_.adjacent_reg_weight);
    //   adjacent_reg_mat_.block<hand_vel_dim_, hand_vel_dim_>(
    //       hand_start_idx + i * hand_vel_dim_, hand_start_idx + (i + 1) * hand_vel_dim_).diagonal().setConstant(
    //           -config_.adjacent_reg_weight);
    // }

    // Set for relative sagittal position between hand and foot
    if (i < config_.motion_len) {
      int foot_idx = i * foot_vel_dim_;
      int hand_idx = hand_start_idx + i * hand_vel_dim_;
      adjacent_reg_mat_(foot_idx, foot_idx) += config_.rel_hand_foot_weight;
      adjacent_reg_mat_(hand_idx, hand_idx) += config_.rel_hand_foot_weight;
      adjacent_reg_mat_(foot_idx, hand_idx) -= config_.rel_hand_foot_weight;
      adjacent_reg_mat_(hand_idx, foot_idx) -= config_.rel_hand_foot_weight;
    }
  }
  // ROS_INFO_STREAM("adjacent_reg_mat_:\n" << adjacent_reg_mat_);
}

void RmapPlanningMulticontact::runOnce(bool publish)
{
  int config_dim = (config_.motion_len + 1) * foot_vel_dim_ + config_.motion_len * hand_vel_dim_;
  int svm_ineq_dim = 3 + 7 * (config_.motion_len - 1);
  int collision_ineq_dim = 0;
  int hand_start_idx = (config_.motion_len + 1) * foot_vel_dim_;

  // Set QP objective matrices
  qp_coeff_.obj_mat_.setZero();
  qp_coeff_.obj_vec_.setZero();
  const FootVelType& start_sample_error =
      sampleError<FootSamplingSpaceType>(identity_foot_sample_, current_foot_sample_seq_.front());
  const FootVelType& target_sample_error =
      sampleError<FootSamplingSpaceType>(target_foot_sample_, current_foot_sample_seq_.back());
  qp_coeff_.obj_mat_.diagonal().template head<foot_vel_dim_>().setConstant(config_.start_foot_weight);
  qp_coeff_.obj_mat_.diagonal().template segment<foot_vel_dim_>(config_.motion_len * foot_vel_dim_).setConstant(1.0);
  qp_coeff_.obj_mat_.diagonal().head(config_dim).array() +=
      start_sample_error.squaredNorm() + target_sample_error.squaredNorm() + config_.reg_weight;
  qp_coeff_.obj_mat_.diagonal().tail(svm_ineq_dim + collision_ineq_dim).head(
      svm_ineq_dim).setConstant(config_.svm_ineq_weight);
  // qp_coeff_.obj_mat_.diagonal().tail(svm_ineq_dim + collision_ineq_dim).tail(
  //     collision_ineq_dim).setConstant(config_.collision_ineq_weight);
  qp_coeff_.obj_vec_.template head<foot_vel_dim_>() = config_.start_foot_weight * start_sample_error;
  qp_coeff_.obj_vec_.template segment<foot_vel_dim_>(config_.motion_len * foot_vel_dim_) = target_sample_error;
  Eigen::VectorXd current_config(config_dim);
  for (int i = 0; i < config_.motion_len + 1; i++) {
    // The implementation of adjacent regularization is not strict because the error between samples is not a simple subtraction
    current_config.template segment<foot_vel_dim_>(i * foot_vel_dim_) =
        sampleError<FootSamplingSpaceType>(identity_foot_sample_, current_foot_sample_seq_[i]);
    if (i < config_.motion_len) {
      current_config.template segment<hand_vel_dim_>(hand_start_idx + i * hand_vel_dim_) =
          sampleError<HandSamplingSpaceType>(identity_hand_sample_, current_hand_sample_seq_[i]);
    }
  }
  // ROS_INFO_STREAM("current_config:\n" << current_config.transpose());
  qp_coeff_.obj_vec_.head(config_dim) += adjacent_reg_mat_ * current_config;
  qp_coeff_.obj_mat_.topLeftCorner(config_dim, config_dim) += adjacent_reg_mat_;

  // Set QP equality matrices of hand contact
  qp_coeff_.eq_mat_.setZero();
  qp_coeff_.eq_vec_.setZero();
  for (int i = 0; i < config_.motion_len; i++) {
    qp_coeff_.eq_mat_(i, hand_start_idx + i * hand_vel_dim_ + 1) = 1;
    qp_coeff_.eq_vec_(i) = config_.hand_lateral_pos - current_hand_sample_seq_[i].y();
  }

  // Set QP inequality matrices of reachability
  qp_coeff_.ineq_mat_.setZero();
  qp_coeff_.ineq_vec_.setZero();
  for (int i = 0; i < config_.motion_len - 1; i++) {
    int ineq_start_idx = 3 + i * 7;
    const FootSampleType& pre_foot_sample = current_foot_sample_seq_[i];
    const FootSampleType& cur_foot_sample = current_foot_sample_seq_[i + 1];
    const FootSampleType& next_foot_sample = current_foot_sample_seq_[i + 2];
    const HandSampleType& pre_hand_sample = current_hand_sample_seq_[i];
    const HandSampleType& cur_hand_sample = current_hand_sample_seq_[i + 1];

    std::shared_ptr<RmapPlanning<FootSamplingSpaceType>> next_foot_rmap_planning;
    if (i % 2 == 0) {
      next_foot_rmap_planning = rmapPlanning<Limb::LeftFoot>();
    } else {
      next_foot_rmap_planning = rmapPlanning<Limb::RightFoot>();
    }
    std::shared_ptr<RmapPlanning<HandSamplingSpaceType>> hand_rmap_planning = rmapPlanning<Limb::LeftHand>();

    const FootSampleType& pre_foot_rel_sample =
        relSample<FootSamplingSpaceType>(cur_foot_sample, pre_foot_sample);
    const FootVelType& pre_foot_rel_svm_grad =
        calcSVMGradWithRmapPlanning<FootSamplingSpaceType>(pre_foot_rel_sample, next_foot_rmap_planning);
    qp_coeff_.ineq_mat_.template block<1, foot_vel_dim_>(ineq_start_idx, (i + 1) * foot_vel_dim_) =
        -1 * pre_foot_rel_svm_grad.transpose() *
        relVelToVelMat<FootSamplingSpaceType>(cur_foot_sample, pre_foot_sample, false);
    qp_coeff_.ineq_mat_.template block<1, foot_vel_dim_>(ineq_start_idx, i * foot_vel_dim_) =
        -1 * pre_foot_rel_svm_grad.transpose() *
        relVelToVelMat<FootSamplingSpaceType>(cur_foot_sample, pre_foot_sample, true);
    qp_coeff_.ineq_vec_.template segment<1>(ineq_start_idx) <<
        calcSVMValueWithRmapPlanning<FootSamplingSpaceType>(pre_foot_rel_sample, next_foot_rmap_planning) - config_.svm_thre;

    const FootSampleType& next_foot_rel_sample =
        relSample<FootSamplingSpaceType>(cur_foot_sample, next_foot_sample);
    const FootVelType& next_foot_rel_svm_grad =
        calcSVMGradWithRmapPlanning<FootSamplingSpaceType>(next_foot_rel_sample, next_foot_rmap_planning);
    qp_coeff_.ineq_mat_.template block<1, foot_vel_dim_>(ineq_start_idx + 1, (i + 1) * foot_vel_dim_) =
        -1 * next_foot_rel_svm_grad.transpose() *
        relVelToVelMat<FootSamplingSpaceType>(cur_foot_sample, next_foot_sample, false);
    qp_coeff_.ineq_mat_.template block<1, foot_vel_dim_>(ineq_start_idx + 1, (i + 2) * foot_vel_dim_) =
        -1 * next_foot_rel_svm_grad.transpose() *
        relVelToVelMat<FootSamplingSpaceType>(cur_foot_sample, next_foot_sample, true);
    qp_coeff_.ineq_vec_.template segment<1>(ineq_start_idx + 1) <<
        calcSVMValueWithRmapPlanning<FootSamplingSpaceType>(next_foot_rel_sample, next_foot_rmap_planning) - config_.svm_thre;

    const HandSampleType& cur_hand_rel_sample = relSampleHandFromFoot(cur_foot_sample, cur_hand_sample, config_.waist_height);
    const HandVelType& cur_hand_rel_svm_grad =
        calcSVMGradWithRmapPlanning<HandSamplingSpaceType>(cur_hand_rel_sample, hand_rmap_planning);
    qp_coeff_.ineq_mat_.template block<1, foot_vel_dim_>(ineq_start_idx + 5, (i + 1) * foot_vel_dim_) =
        -1 * cur_hand_rel_svm_grad.transpose() *
        relSampleGradHandFromFoot(cur_foot_sample, cur_hand_sample, false);
    qp_coeff_.ineq_mat_.template block<1, hand_vel_dim_>(ineq_start_idx + 5, hand_start_idx + (i + 1) * hand_vel_dim_) =
        -1 * cur_hand_rel_svm_grad.transpose() *
        relSampleGradHandFromFoot(cur_foot_sample, cur_hand_sample, true);
    qp_coeff_.ineq_vec_.template segment<1>(ineq_start_idx + 5) <<
        calcSVMValueWithRmapPlanning<HandSamplingSpaceType>(cur_hand_rel_sample, hand_rmap_planning) - config_.svm_thre;
  }
  qp_coeff_.ineq_mat_.rightCols(svm_ineq_dim + collision_ineq_dim).diagonal().head(svm_ineq_dim).setConstant(-1);

  // ROS_INFO_STREAM("qp_coeff_.obj_mat_:\n" << qp_coeff_.obj_mat_);
  // ROS_INFO_STREAM("qp_coeff_.obj_vec_:\n" << qp_coeff_.obj_vec_.transpose());
  // ROS_INFO_STREAM("qp_coeff_.eq_mat_:\n" << qp_coeff_.eq_mat_);
  // ROS_INFO_STREAM("qp_coeff_.eq_vec_:\n" << qp_coeff_.eq_vec_.transpose());
  // ROS_INFO_STREAM("qp_coeff_.ineq_mat_:\n" << qp_coeff_.ineq_mat_);
  // ROS_INFO_STREAM("qp_coeff_.ineq_vec_:\n" << qp_coeff_.ineq_vec_.transpose());

  // Solve QP
  Eigen::VectorXd vel_all = qp_solver_->solve(qp_coeff_);
  if (qp_solver_->solve_failed_) {
    vel_all.setZero();
  }

  // Integrate
  for (int i = 0; i < config_.motion_len + 1; i++) {
    integrateVelToSample<FootSamplingSpaceType>(
        current_foot_sample_seq_[i], vel_all.template segment<foot_vel_dim_>(i * foot_vel_dim_));
    if (i < config_.motion_len) {
      integrateVelToSample<HandSamplingSpaceType>(
          current_hand_sample_seq_[i], vel_all.template segment<hand_vel_dim_>(hand_start_idx + i * hand_vel_dim_));
    }
  }

  if (publish) {
    // Publish
    publishMarkerArray();
    publishCurrentState();
  }
}

void RmapPlanningMulticontact::runLoop()
{
  setup();

  ros::Rate rate(config_.loop_rate);
  int loop_idx = 0;
  while (ros::ok()) {
    runOnce(loop_idx % config_.publish_interval == 0);

    rate.sleep();
    ros::spinOnce();
    loop_idx++;
  }
}

void RmapPlanningMulticontact::publishMarkerArray() const
{
}
// {
//   std_msgs::Header header_msg;
//   header_msg.frame_id = "world";
//   header_msg.stamp = ros::Time::now();

//   // Instantiate marker array
//   visualization_msgs::MarkerArray marker_arr_msg;

//   // Delete marker
//   visualization_msgs::Marker del_marker;
//   del_marker.action = visualization_msgs::Marker::DELETEALL;
//   del_marker.header = header_msg;
//   del_marker.id = marker_arr_msg.markers.size();
//   marker_arr_msg.markers.push_back(del_marker);

//   // XY plane marker
//   visualization_msgs::Marker xy_plane_marker;
//   double plane_thickness = 0.01;
//   xy_plane_marker.header = header_msg;
//   xy_plane_marker.ns = "xy_plane";
//   xy_plane_marker.id = marker_arr_msg.markers.size();
//   xy_plane_marker.type = visualization_msgs::Marker::CUBE;
//   xy_plane_marker.color = OmgCore::toColorRGBAMsg({0.8, 0.8, 0.8, 1.0});
//   xy_plane_marker.scale.x = 100.0;
//   xy_plane_marker.scale.y = 100.0;
//   xy_plane_marker.scale.z = plane_thickness;
//   xy_plane_marker.pose = OmgCore::toPoseMsg(
//       sva::PTransformd(Eigen::Vector3d(0, 0, config_.svm_thre - 0.5 * plane_thickness)));
//   marker_arr_msg.markers.push_back(xy_plane_marker);

//   // Reachable grids marker
//   if (grid_set_msg_) {
//     visualization_msgs::Marker grids_marker;
//     SampleType sample_range = sample_max_ - sample_min_;
//     grids_marker.header = header_msg;
//     grids_marker.ns = "reachable_grids";
//     grids_marker.id = marker_arr_msg.markers.size();
//     grids_marker.type = visualization_msgs::Marker::CUBE_LIST;
//     grids_marker.color = OmgCore::toColorRGBAMsg({0.8, 0.0, 0.0, 0.5});
//     grids_marker.scale = OmgCore::toVector3Msg(
//         calcGridCubeScale<SamplingSpaceType>(grid_set_msg_->divide_nums, sample_range));
//     grids_marker.pose = OmgCore::toPoseMsg(sva::PTransformd::Identity());
//     loopGrid<SamplingSpaceType>(
//         grid_set_msg_->divide_nums,
//         sample_min_,
//         sample_range,
//         [&](int grid_idx, const SampleType& sample) {
//           if (grid_set_msg_->values[grid_idx] > config_.svm_thre) {
//             grids_marker.points.push_back(
//                 OmgCore::toPointMsg(sampleToCloudPos<SamplingSpaceType>(sample)));
//           }
//         });
//     marker_arr_msg.markers.push_back(grids_marker);
//   }

//   marker_arr_pub_.publish(marker_arr_msg);
// }

void RmapPlanningMulticontact::publishCurrentState() const
{
  std_msgs::Header header_msg;
  header_msg.frame_id = "world";
  header_msg.stamp = ros::Time::now();

  // Publish pose array
  geometry_msgs::PoseArray pose_arr_msg;
  pose_arr_msg.header = header_msg;
  pose_arr_msg.poses.resize(2 * config_.motion_len + 1);
  for (int i = 0; i < config_.motion_len + 1; i++) {
    pose_arr_msg.poses[i] = OmgCore::toPoseMsg(sampleToPose<FootSamplingSpaceType>(current_foot_sample_seq_[i]));
    if (i < config_.motion_len) {
      pose_arr_msg.poses[config_.motion_len + 1 + i] =
          OmgCore::toPoseMsg(sampleToPose<HandSamplingSpaceType>(current_hand_sample_seq_[i]));
    }
  }
  current_pose_arr_pub_.publish(pose_arr_msg);

  // Publish polygon array
  jsk_recognition_msgs::PolygonArray poly_arr_msg;
  poly_arr_msg.header = header_msg;
  poly_arr_msg.polygons.resize(config_.motion_len + 1);
  for (int i = 0; i < config_.motion_len + 1; i++) {
    poly_arr_msg.polygons[i].header = header_msg;
    sva::PTransformd foot_pose = sampleToPose<FootSamplingSpaceType>(current_foot_sample_seq_[i]);
    poly_arr_msg.polygons[i].polygon.points.resize(config_.foot_vertices.size());
    for (size_t j = 0; j < config_.foot_vertices.size(); j++) {
      poly_arr_msg.polygons[i].polygon.points[j] =
          OmgCore::toPoint32Msg(foot_pose.rotation().transpose() * config_.foot_vertices[j] + foot_pose.translation());
    }
  }
  current_poly_arr_pub_.publish(poly_arr_msg);
}

void RmapPlanningMulticontact::transCallback(
    const geometry_msgs::TransformStamped::ConstPtr& trans_st_msg)
{
  if (trans_st_msg->child_frame_id == "target") {
    target_foot_sample_ = poseToSample<FootSamplingSpaceType>(OmgCore::toSvaPTransform(trans_st_msg->transform));
  }
}
