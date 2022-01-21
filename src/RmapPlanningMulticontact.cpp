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
}
// {
//   qp_coeff_.setup(vel_dim_, 0, 1);
//   qp_coeff_.x_min_.setConstant(-config_.delta_config_limit);
//   qp_coeff_.x_max_.setConstant(config_.delta_config_limit);

//   qp_solver_ = OmgCore::allocateQpSolver(OmgCore::QpSolverType::JRLQP);

//   current_sample_ = poseToSample<SamplingSpaceType>(config_.initial_sample_pose);
// }

void RmapPlanningMulticontact::runOnce(bool publish)
{
}
// {
//   // Set QP coefficients
//   qp_coeff_.obj_vec_ = sampleError<SamplingSpaceType>(target_sample_, current_sample_);
//   double lambda = qp_coeff_.obj_vec_.squaredNorm() + 1e-3;
//   qp_coeff_.obj_mat_.diagonal().setConstant(1.0 + lambda);
//   qp_coeff_.ineq_mat_ = -1 * calcSVMGrad<SamplingSpaceType>(
//       current_sample_, svm_mo_->param, svm_mo_, svm_coeff_vec_, svm_sv_mat_).transpose();
//   qp_coeff_.ineq_vec_ << calcSVMValue<SamplingSpaceType>(
//       current_sample_, svm_mo_->param, svm_mo_, svm_coeff_vec_, svm_sv_mat_) - config_.svm_thre;

//   // Solve QP
//   const VelType& vel = qp_solver_->solve(qp_coeff_);

//   // Integrate
//   integrateVelToSample<SamplingSpaceType>(current_sample_, vel);

//   if (publish) {
//     // Publish
//     publishMarkerArray();
//     publishCurrentState();
//   }
// }

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
}
// {
//   std_msgs::Header header_msg;
//   header_msg.frame_id = "world";
//   header_msg.stamp = ros::Time::now();

//   // Publish point
//   geometry_msgs::PointStamped pos_msg;
//   pos_msg.header = header_msg;
//   pos_msg.point = OmgCore::toPointMsg(sampleToCloudPos<SamplingSpaceType>(current_sample_));
//   current_pos_pub_.publish(pos_msg);

//   // Publish pose
//   geometry_msgs::PoseStamped pose_msg;
//   pose_msg.header = header_msg;
//   pose_msg.pose = OmgCore::toPoseMsg(sampleToPose<SamplingSpaceType>(current_sample_));
//   current_pose_pub_.publish(pose_msg);
// }

void RmapPlanningMulticontact::transCallback(
    const geometry_msgs::TransformStamped::ConstPtr& trans_st_msg)
{
  if (trans_st_msg->child_frame_id == "target") {
    target_sample_ = poseToSample<FootSamplingSpaceType>(OmgCore::toSvaPTransform(trans_st_msg->transform));
  }
}
