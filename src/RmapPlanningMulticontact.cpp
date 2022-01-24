/* Author: Masaki Murooka */

#include <chrono>

#include <mc_rtc/constants.h>

#include <geometry_msgs/PoseArray.h>
#include <sensor_msgs/PointCloud.h>
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
  current_left_poly_arr_pub_ = nh_.template advertise<jsk_recognition_msgs::PolygonArray>(
      "current_left_poly_arr", 1, true);
  current_right_poly_arr_pub_ = nh_.template advertise<jsk_recognition_msgs::PolygonArray>(
      "current_right_poly_arr", 1, true);
  current_cloud_pub_ = nh_.template advertise<sensor_msgs::PointCloud>(
      "current_cloud", 1, true);

  rmap_planning_list_[Limb::LeftFoot] =
      std::make_shared<RmapPlanning<FootSamplingSpaceType>>(
          svm_path_list.at(Limb::LeftFoot),
          bag_path_list.at(Limb::LeftFoot),
          false);
  rmap_planning_list_[Limb::RightFoot] =
      std::make_shared<RmapPlanning<FootSamplingSpaceType>>(
          svm_path_list.at(Limb::RightFoot),
          bag_path_list.at(Limb::RightFoot),
          false);
  rmap_planning_list_[Limb::LeftHand] =
      std::make_shared<RmapPlanning<HandSamplingSpaceType>>(
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
  if (config_.motion_len % 2 != 0) {
    mc_rtc::log::error_and_throw<std::runtime_error>(
        "[RmapPlanningMulticontact::setup] motion_len must be a multiple of 2, but is {}",
        config_.motion_len);
  }

  // Setup dimensions
  foot_num_ = config_.motion_len + 1;
  hand_num_ = config_.motion_len / 2;
  config_dim_ = foot_num_ * foot_vel_dim_ + hand_num_ * hand_vel_dim_;
  svm_ineq_dim_ = (foot_num_ - 1) + (4 * hand_num_ - 1);
  collision_ineq_dim_ = 0;
  hand_start_config_idx_ = foot_num_ * foot_vel_dim_;

  // Setup QP coefficients and solver
  // Introduce variables for inequality constraint errors
  qp_coeff_.setup(
      config_dim_ + svm_ineq_dim_ + collision_ineq_dim_,
      hand_num_,
      svm_ineq_dim_ + collision_ineq_dim_);
  qp_coeff_.x_min_.head(config_dim_).setConstant(-config_.delta_config_limit);
  qp_coeff_.x_max_.head(config_dim_).setConstant(config_.delta_config_limit);
  qp_coeff_.x_min_.tail(svm_ineq_dim_ + collision_ineq_dim_).setConstant(-1e10);
  qp_coeff_.x_max_.tail(svm_ineq_dim_ + collision_ineq_dim_).setConstant(1e10);

  qp_solver_ = OmgCore::allocateQpSolver(OmgCore::QpSolverType::JRLQP);

  // Setup current sample sequence
  current_foot_sample_seq_.resize(foot_num_);
  current_hand_sample_seq_.resize(hand_num_);
  for (int i = 0; i < foot_num_; i++) {
    current_foot_sample_seq_[i] = poseToSample<FootSamplingSpaceType>(
        config_.initial_sample_pose_list.at(i % 2 == 0 ? Limb::LeftFoot : Limb::RightFoot));
  }
  for (int i = 0; i < hand_num_; i++) {
    current_hand_sample_seq_[i] = poseToSample<HandSamplingSpaceType>(
        config_.initial_sample_pose_list.at(Limb::LeftHand));
  }
  start_foot_sample_ = poseToSample<FootSamplingSpaceType>(config_.initial_sample_pose_list.at(Limb::LeftFoot));

  // Setup adjacent regularization
  adjacent_reg_mat_.setZero(config_dim_, config_dim_);
  //// Set for adjacent foot
  for (int i = 0; i < foot_num_; i++) {
    adjacent_reg_mat_.block<foot_vel_dim_, foot_vel_dim_>(
        i * foot_vel_dim_, i * foot_vel_dim_).diagonal().setConstant(
            ((i == 0 || i == foot_num_ - 1) ? 1 : 2) * config_.adjacent_reg_weight);
    if (i < foot_num_ - 1) {
      adjacent_reg_mat_.block<foot_vel_dim_, foot_vel_dim_>(
          (i + 1) * foot_vel_dim_, i * foot_vel_dim_).diagonal().setConstant(
              -config_.adjacent_reg_weight);
      adjacent_reg_mat_.block<foot_vel_dim_, foot_vel_dim_>(
          i * foot_vel_dim_, (i + 1) * foot_vel_dim_).diagonal().setConstant(
              -config_.adjacent_reg_weight);
    }
  }
  //// Set for relative sagittal position between hand and foot
  for (int i = 0; i < foot_num_ - 1; i++) {
    int foot_config_idx = i * foot_vel_dim_;
    int hand_config_idx = hand_start_config_idx_ + (i % 2 == 0 ? i / 2 : (i - 1) / 2) * hand_vel_dim_;
    adjacent_reg_mat_(foot_config_idx, foot_config_idx) += config_.rel_hand_foot_weight;
    adjacent_reg_mat_(hand_config_idx, hand_config_idx) += config_.rel_hand_foot_weight;
    adjacent_reg_mat_(foot_config_idx, hand_config_idx) -= config_.rel_hand_foot_weight;
    adjacent_reg_mat_(hand_config_idx, foot_config_idx) -= config_.rel_hand_foot_weight;
  }
  // ROS_INFO_STREAM("adjacent_reg_mat_:\n" << adjacent_reg_mat_);
}

void RmapPlanningMulticontact::runOnce(bool publish)
{
  // Set QP objective matrices
  qp_coeff_.obj_mat_.setZero();
  qp_coeff_.obj_vec_.setZero();
  const FootVelType& start_sample_error =
      sampleError<FootSamplingSpaceType>(start_foot_sample_, current_foot_sample_seq_.front());
  const FootVelType& target_sample_error =
      sampleError<FootSamplingSpaceType>(target_foot_sample_, current_foot_sample_seq_.back());
  qp_coeff_.obj_mat_.diagonal().template head<foot_vel_dim_>().setConstant(config_.start_foot_weight);
  qp_coeff_.obj_mat_.diagonal().template segment<foot_vel_dim_>((foot_num_ - 1) * foot_vel_dim_).setConstant(1.0);
  qp_coeff_.obj_mat_.diagonal().head(config_dim_).array() +=
      start_sample_error.squaredNorm() + target_sample_error.squaredNorm() + config_.reg_weight;
  qp_coeff_.obj_mat_.diagonal().tail(svm_ineq_dim_ + collision_ineq_dim_).head(
      svm_ineq_dim_).setConstant(config_.svm_ineq_weight);
  // qp_coeff_.obj_mat_.diagonal().tail(svm_ineq_dim_ + collision_ineq_dim_).tail(
  //     collision_ineq_dim_).setConstant(config_.collision_ineq_weight);
  qp_coeff_.obj_vec_.template head<foot_vel_dim_>() = config_.start_foot_weight * start_sample_error;
  qp_coeff_.obj_vec_.template segment<foot_vel_dim_>((foot_num_ - 1) * foot_vel_dim_) = target_sample_error;
  Eigen::VectorXd current_config(config_dim_);
  // This implementation of adjacent regularization is not exact because the error between samples is not a simple subtraction
  for (int i = 0; i < foot_num_; i++) {
    current_config.template segment<foot_vel_dim_>(i * foot_vel_dim_) =
        sampleError<FootSamplingSpaceType>(identity_foot_sample_, current_foot_sample_seq_[i]);
  }
  for (int i = 0; i < hand_num_; i++) {
    current_config.template segment<hand_vel_dim_>(hand_start_config_idx_ + i * hand_vel_dim_) =
        sampleError<HandSamplingSpaceType>(identity_hand_sample_, current_hand_sample_seq_[i]);
  }
  // ROS_INFO_STREAM("current_config:\n" << current_config.transpose());
  qp_coeff_.obj_vec_.head(config_dim_) += adjacent_reg_mat_ * current_config;
  qp_coeff_.obj_mat_.topLeftCorner(config_dim_, config_dim_) += adjacent_reg_mat_;

  // Set QP equality matrices of hand contact
  qp_coeff_.eq_mat_.setZero();
  qp_coeff_.eq_vec_.setZero();
  for (int i = 0; i < hand_num_; i++) {
    qp_coeff_.eq_mat_(i, hand_start_config_idx_ + i * hand_vel_dim_ + 1) = 1;
    qp_coeff_.eq_vec_(i) = config_.hand_lateral_pos - current_hand_sample_seq_[i].y();
  }

  // Set QP inequality matrices of reachability
  qp_coeff_.ineq_mat_.setZero();
  qp_coeff_.ineq_vec_.setZero();
  //// Set for reachability between foot
  for (int i = 0; i < foot_num_ - 1; i++) {
    const FootSampleType& pre_foot_sample = current_foot_sample_seq_[i];
    const FootSampleType& suc_foot_sample = current_foot_sample_seq_[i + 1];
    std::shared_ptr<RmapPlanning<FootSamplingSpaceType>> rmap_planning =
        i % 2 == 0 ? rmapPlanning<Limb::RightFoot>() : rmapPlanning<Limb::LeftFoot>();

    const FootSampleType& rel_sample =
        relSample<FootSamplingSpaceType>(pre_foot_sample, suc_foot_sample);
    const FootVelType& rel_svm_grad = rmap_planning->calcSVMGrad(rel_sample);
    qp_coeff_.ineq_mat_.template block<1, foot_vel_dim_>(i, i * foot_vel_dim_) =
        -1 * rel_svm_grad.transpose() *
        relVelToVelMat<FootSamplingSpaceType>(pre_foot_sample, suc_foot_sample, false);
    qp_coeff_.ineq_mat_.template block<1, foot_vel_dim_>(i, (i + 1) * foot_vel_dim_) =
        -1 * rel_svm_grad.transpose() *
        relVelToVelMat<FootSamplingSpaceType>(pre_foot_sample, suc_foot_sample, true);
    qp_coeff_.ineq_vec_.template segment<1>(i) <<
        rmap_planning->calcSVMValue(rel_sample) - config_.svm_thre;
  }
  //// Set for reachability from foot to hand
  for (int i = 0; i < hand_num_; i++) {
    int start_ineq_idx = foot_num_ - 1 + 4 * i - 1;
    const FootSampleType& pre1_foot_sample = current_foot_sample_seq_[2 * i];
    const FootSampleType& suc1_foot_sample = current_foot_sample_seq_[2 * i + 1];
    const FootSampleType& suc2_foot_sample = current_foot_sample_seq_[2 * i + 2];
    const HandSampleType& hand_sample = current_hand_sample_seq_[i];
    std::shared_ptr<RmapPlanning<HandSamplingSpaceType>> rmap_planning = rmapPlanning<Limb::LeftHand>();

    if (i != 0) {
      const FootSampleType& pre2_foot_sample = current_foot_sample_seq_[2 * i - 1];
      const FootSampleType& pre12_foot_sample =
          midSample<FootSamplingSpaceType>(pre1_foot_sample, pre2_foot_sample);
      const HandSampleType& pre12_rel_sample =
          relSampleHandFromFoot(pre12_foot_sample, hand_sample, config_.waist_height);
      const HandVelType& pre12_rel_svm_grad = rmap_planning->calcSVMGrad(pre12_rel_sample);
      // The implementation of gradient of mean sample is not exact because the mean of two samples is not a simple arithmetic mean
      Eigen::MatrixXd pre12_foot_ineq_mat =
          -1 * pre12_rel_svm_grad.transpose() * relSampleGradHandFromFoot(pre12_foot_sample, hand_sample, false) / 2;
      qp_coeff_.ineq_mat_.template block<1, foot_vel_dim_>(start_ineq_idx + 0, (2 * i - 1) * foot_vel_dim_) =
          pre12_foot_ineq_mat;
      qp_coeff_.ineq_mat_.template block<1, foot_vel_dim_>(start_ineq_idx + 0, (2 * i) * foot_vel_dim_) =
          pre12_foot_ineq_mat;
      qp_coeff_.ineq_mat_.template block<1, hand_vel_dim_>(start_ineq_idx + 0, hand_start_config_idx_ + i * hand_vel_dim_) =
          -1 * pre12_rel_svm_grad.transpose() *
          relSampleGradHandFromFoot(pre12_foot_sample, hand_sample, true);
      qp_coeff_.ineq_vec_.template segment<1>(start_ineq_idx + 0) <<
          rmap_planning->calcSVMValue(pre12_rel_sample) - config_.svm_thre;
    }

    const HandSampleType& pre1_rel_sample =
        relSampleHandFromFoot(pre1_foot_sample, hand_sample, config_.waist_height);
    const HandVelType& pre1_rel_svm_grad = rmap_planning->calcSVMGrad(pre1_rel_sample);
    qp_coeff_.ineq_mat_.template block<1, foot_vel_dim_>(start_ineq_idx + 1, (2 * i) * foot_vel_dim_) =
        -1 * pre1_rel_svm_grad.transpose() *
        relSampleGradHandFromFoot(pre1_foot_sample, hand_sample, false);
    qp_coeff_.ineq_mat_.template block<1, hand_vel_dim_>(start_ineq_idx + 1, hand_start_config_idx_ + i * hand_vel_dim_) =
        -1 * pre1_rel_svm_grad.transpose() *
        relSampleGradHandFromFoot(pre1_foot_sample, hand_sample, true);
    qp_coeff_.ineq_vec_.template segment<1>(start_ineq_idx + 1) <<
        rmap_planning->calcSVMValue(pre1_rel_sample) - config_.svm_thre;

    const HandSampleType& suc1_rel_sample =
        relSampleHandFromFoot(suc1_foot_sample, hand_sample, config_.waist_height);
    const HandVelType& suc1_rel_svm_grad = rmap_planning->calcSVMGrad(suc1_rel_sample);
    qp_coeff_.ineq_mat_.template block<1, foot_vel_dim_>(start_ineq_idx + 2, (2 * i + 1) * foot_vel_dim_) =
        -1 * suc1_rel_svm_grad.transpose() *
        relSampleGradHandFromFoot(suc1_foot_sample, hand_sample, false);
    qp_coeff_.ineq_mat_.template block<1, hand_vel_dim_>(start_ineq_idx + 2, hand_start_config_idx_ + i * hand_vel_dim_) =
        -1 * suc1_rel_svm_grad.transpose() *
        relSampleGradHandFromFoot(suc1_foot_sample, hand_sample, true);
    qp_coeff_.ineq_vec_.template segment<1>(start_ineq_idx + 2) <<
        rmap_planning->calcSVMValue(suc1_rel_sample) - config_.svm_thre;

    const FootSampleType& suc12_foot_sample =
        midSample<FootSamplingSpaceType>(suc1_foot_sample, suc2_foot_sample);
    const HandSampleType& suc12_rel_sample =
        relSampleHandFromFoot(suc12_foot_sample, hand_sample, config_.waist_height);
    const HandVelType& suc12_rel_svm_grad = rmap_planning->calcSVMGrad(suc12_rel_sample);
    Eigen::MatrixXd suc12_foot_ineq_mat =
        -1 * suc12_rel_svm_grad.transpose() * relSampleGradHandFromFoot(suc12_foot_sample, hand_sample, false) / 2;
    qp_coeff_.ineq_mat_.template block<1, foot_vel_dim_>(start_ineq_idx + 3, (2 * i + 1) * foot_vel_dim_) =
        suc12_foot_ineq_mat;
    qp_coeff_.ineq_mat_.template block<1, foot_vel_dim_>(start_ineq_idx + 3, (2 * i + 2) * foot_vel_dim_) =
        suc12_foot_ineq_mat;
    qp_coeff_.ineq_mat_.template block<1, hand_vel_dim_>(start_ineq_idx + 3, hand_start_config_idx_ + i * hand_vel_dim_) =
        -1 * suc12_rel_svm_grad.transpose() *
        relSampleGradHandFromFoot(suc12_foot_sample, hand_sample, true);
    qp_coeff_.ineq_vec_.template segment<1>(start_ineq_idx + 3) <<
        rmap_planning->calcSVMValue(suc12_rel_sample) - config_.svm_thre;
  }
  qp_coeff_.ineq_mat_.rightCols(
      svm_ineq_dim_ + collision_ineq_dim_).diagonal().head(svm_ineq_dim_).setConstant(-1);

  // Set QP variables limit
  for (int i = 0; i < foot_num_; i++) {
    qp_coeff_.x_min_.segment(i * foot_vel_dim_, foot_vel_dim_) =
        (config_.foot_pos_limits.first - current_foot_sample_seq_[i]).cwiseMax(-config_.delta_config_limit);
    qp_coeff_.x_max_.segment(i * foot_vel_dim_, foot_vel_dim_) =
        (config_.foot_pos_limits.second - current_foot_sample_seq_[i]).cwiseMin(config_.delta_config_limit);
  }
  for (int i = 0; i < hand_num_; i++) {
    qp_coeff_.x_min_.segment(hand_start_config_idx_ + i * hand_vel_dim_, hand_vel_dim_) =
        (config_.hand_pos_limits.first - current_hand_sample_seq_[i]).cwiseMax(-config_.delta_config_limit);
    qp_coeff_.x_max_.segment(hand_start_config_idx_ + i * hand_vel_dim_, hand_vel_dim_) =
        (config_.hand_pos_limits.second - current_hand_sample_seq_[i]).cwiseMin(config_.delta_config_limit);
  }

  // ROS_INFO_STREAM("qp_coeff_.obj_mat_:\n" << qp_coeff_.obj_mat_);
  // ROS_INFO_STREAM("qp_coeff_.obj_vec_:\n" << qp_coeff_.obj_vec_.transpose());
  // ROS_INFO_STREAM("qp_coeff_.eq_mat_:\n" << qp_coeff_.eq_mat_);
  // ROS_INFO_STREAM("qp_coeff_.eq_vec_:\n" << qp_coeff_.eq_vec_.transpose());
  // ROS_INFO_STREAM("qp_coeff_.ineq_mat_:\n" << qp_coeff_.ineq_mat_);
  // ROS_INFO_STREAM("qp_coeff_.ineq_vec_:\n" << qp_coeff_.ineq_vec_.transpose());
  // ROS_INFO_STREAM("qp_coeff_.x_min_:\n" << qp_coeff_.x_min_.transpose());
  // ROS_INFO_STREAM("qp_coeff_.x_max_:\n" << qp_coeff_.x_max_.transpose());

  // Solve QP
  Eigen::VectorXd vel_all = qp_solver_->solve(qp_coeff_);
  if (qp_solver_->solve_failed_) {
    vel_all.setZero();
  }

  // Integrate
  for (int i = 0; i < foot_num_; i++) {
    integrateVelToSample<FootSamplingSpaceType>(
        current_foot_sample_seq_[i], vel_all.template segment<foot_vel_dim_>(i * foot_vel_dim_));
  }
  for (int i = 0; i < hand_num_; i++) {
    integrateVelToSample<HandSamplingSpaceType>(
        current_hand_sample_seq_[i], vel_all.template segment<hand_vel_dim_>(hand_start_config_idx_ + i * hand_vel_dim_));
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

  // Foot reachable grids marker
  {
    visualization_msgs::Marker grids_marker;
    grids_marker.header = header_msg;
    grids_marker.type = visualization_msgs::Marker::CUBE_LIST;
    grids_marker.color = OmgCore::toColorRGBAMsg({0.8, 0.0, 0.0, 0.3});

    for (int i = 1; i < foot_num_; i++) {
      std::shared_ptr<RmapPlanning<FootSamplingSpaceType>> rmap_planning =
          i % 2 == 0 ? rmapPlanning<Limb::LeftFoot>() : rmapPlanning<Limb::RightFoot>();
      const FootSampleType& sample_min = rmap_planning->sample_min_;
      const FootSampleType& sample_max = rmap_planning->sample_max_;
      const FootSampleType& sample_range = sample_max - sample_min;
      const auto& grid_set_msg = rmap_planning->grid_set_msg_;

      grids_marker.ns = "foot_reachable_grids_" + std::to_string(i);
      grids_marker.id = marker_arr_msg.markers.size();
      grids_marker.scale = OmgCore::toVector3Msg(
          calcGridCubeScale<FootSamplingSpaceType>(grid_set_msg->divide_nums, sample_range));
      grids_marker.scale.z = 0.01;
      grids_marker.pose = OmgCore::toPoseMsg(sampleToPose<FootSamplingSpaceType>(current_foot_sample_seq_[i - 1]));
      grids_marker.color = OmgCore::toColorRGBAMsg(
          i % 2 == 0 ? std::array<double, 4>{0.8, 0.0, 0.0, 0.3} : std::array<double, 4>{0.0, 0.8, 0.0, 0.3});
      const FootSampleType& slice_sample =
          relSample<FootSamplingSpaceType>(current_foot_sample_seq_[i - 1], current_foot_sample_seq_[i]);
      GridIdxsType<FootSamplingSpaceType> slice_divide_idxs;
      gridDivideRatiosToIdxs(
          slice_divide_idxs,
          (slice_sample - sample_min).array() / sample_range.array(),
          grid_set_msg->divide_nums);
      grids_marker.points.clear();
      loopGrid<FootSamplingSpaceType>(
          grid_set_msg->divide_nums,
          sample_min,
          sample_range,
          [&](int grid_idx, const FootSampleType& sample) {
            if (grid_set_msg->values[grid_idx] > config_.svm_thre) {
              Eigen::Vector3d pos = sampleToCloudPos<FootSamplingSpaceType>(sample);
              pos.z() = 0;
              grids_marker.points.push_back(OmgCore::toPointMsg(pos));
            }
          },
          std::vector<int>{0, 1},
          slice_divide_idxs);
      marker_arr_msg.markers.push_back(grids_marker);
    }
  }

  // Hand reachable grids marker
  {
    std::shared_ptr<RmapPlanning<HandSamplingSpaceType>> rmap_planning = rmapPlanning<Limb::LeftHand>();
    const HandSampleType& sample_min = rmap_planning->sample_min_;
    const HandSampleType& sample_max = rmap_planning->sample_max_;
    const HandSampleType& sample_range = sample_max - sample_min;
    const auto& grid_set_msg = rmap_planning->grid_set_msg_;

    visualization_msgs::Marker grids_marker;
    grids_marker.header = header_msg;
    grids_marker.type = visualization_msgs::Marker::CUBE_LIST;
    grids_marker.color = OmgCore::toColorRGBAMsg({0.0, 0.0, 0.8, 0.1});
    grids_marker.scale = OmgCore::toVector3Msg(
        calcGridCubeScale<HandSamplingSpaceType>(grid_set_msg->divide_nums, sample_range));
    loopGrid<HandSamplingSpaceType>(
        grid_set_msg->divide_nums,
        sample_min,
        sample_range,
        [&](int grid_idx, const HandSampleType& sample) {
          if (grid_set_msg->values[grid_idx] > config_.svm_thre) {
            grids_marker.points.push_back(
                OmgCore::toPointMsg(sampleToCloudPos<HandSamplingSpaceType>(sample)));
          }
        });
    for (int i = 0; i < foot_num_ - 1; i++) {
      // Publish only the grid set at the timing of hand transition
      if (i % 2 == 0) {
        continue;
      }
      grids_marker.ns = "hand_reachable_grids_" + std::to_string(i);
      sva::PTransformd pose = sampleToPose<FootSamplingSpaceType>(
          midSample<FootSamplingSpaceType>(current_foot_sample_seq_[i], current_foot_sample_seq_[i + 1]));
      pose.translation().z() = config_.waist_height;
      grids_marker.pose = OmgCore::toPoseMsg(pose);
      grids_marker.id = marker_arr_msg.markers.size();
      marker_arr_msg.markers.push_back(grids_marker);
    }
  }

  marker_arr_pub_.publish(marker_arr_msg);
}

void RmapPlanningMulticontact::publishCurrentState() const
{
  std_msgs::Header header_msg;
  header_msg.frame_id = "world";
  header_msg.stamp = ros::Time::now();

  // Publish pose array for foot and hand
  geometry_msgs::PoseArray pose_arr_msg;
  pose_arr_msg.header = header_msg;
  pose_arr_msg.poses.resize(foot_num_ + hand_num_);
  for (int i = 0; i < foot_num_; i++) {
    pose_arr_msg.poses[i] =
        OmgCore::toPoseMsg(sampleToPose<FootSamplingSpaceType>(current_foot_sample_seq_[i]));
  }
  for (int i = 0; i < hand_num_; i++) {
    pose_arr_msg.poses[foot_num_ + i] =
        OmgCore::toPoseMsg(sampleToPose<HandSamplingSpaceType>(current_hand_sample_seq_[i]));
  }
  current_pose_arr_pub_.publish(pose_arr_msg);

  // Publish polygon array for foot
  jsk_recognition_msgs::PolygonArray poly_arr_msg;
  jsk_recognition_msgs::PolygonArray left_poly_arr_msg;
  jsk_recognition_msgs::PolygonArray right_poly_arr_msg;
  poly_arr_msg.header = header_msg;
  left_poly_arr_msg.header = header_msg;
  right_poly_arr_msg.header = header_msg;
  poly_arr_msg.polygons.resize(foot_num_);
  for (int i = 0; i < foot_num_; i++) {
    poly_arr_msg.polygons[i].header = header_msg;
    const sva::PTransformd& foot_pose = sampleToPose<FootSamplingSpaceType>(current_foot_sample_seq_[i]);
    poly_arr_msg.polygons[i].polygon.points.resize(config_.foot_vertices.size());
    for (size_t j = 0; j < config_.foot_vertices.size(); j++) {
      poly_arr_msg.polygons[i].polygon.points[j] =
          OmgCore::toPoint32Msg(foot_pose.rotation().transpose() * config_.foot_vertices[j] + foot_pose.translation());
    }
    if (i % 2 == 0) {
      left_poly_arr_msg.polygons.push_back(poly_arr_msg.polygons[i]);
    } else {
      right_poly_arr_msg.polygons.push_back(poly_arr_msg.polygons[i]);
    }
  }
  current_poly_arr_pub_.publish(poly_arr_msg);
  current_left_poly_arr_pub_.publish(left_poly_arr_msg);
  current_right_poly_arr_pub_.publish(right_poly_arr_msg);

  // Publish cloud for hand
  sensor_msgs::PointCloud cloud_msg;
  cloud_msg.header = header_msg;
  for (int i = 0; i < hand_num_; i++) {
    cloud_msg.points.push_back(OmgCore::toPoint32Msg(
        sampleToCloudPos<HandSamplingSpaceType>(current_hand_sample_seq_[i])));
  }
  current_cloud_pub_.publish(cloud_msg);
}

void RmapPlanningMulticontact::transCallback(
    const geometry_msgs::TransformStamped::ConstPtr& trans_st_msg)
{
  if (trans_st_msg->child_frame_id == "target") {
    target_foot_sample_ = poseToSample<FootSamplingSpaceType>(OmgCore::toSvaPTransform(trans_st_msg->transform));
  }
}
