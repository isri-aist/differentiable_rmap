/* Author: Masaki Murooka */

#include <chrono>

#include <mc_rtc/constants.h>

#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <visualization_msgs/MarkerArray.h>

#include <optmotiongen/Utils/RosUtils.h>
#include <optmotiongen/Utils/QpUtils.h>

#include <differentiable_rmap/RmapPlanning.h>
#include <differentiable_rmap/SVMUtils.h>
#include <differentiable_rmap/libsvm_hotfix.h>

using namespace DiffRmap;


template <SamplingSpace SamplingSpaceType>
RmapPlanning<SamplingSpaceType>::RmapPlanning(const std::string& svm_path)
{
  // Setup ROS
  trans_sub_ = nh_.subscribe("interactive_marker_transform", 1, &RmapPlanning<SamplingSpaceType>::transCallback, this);
  marker_arr_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("marker_arr", 1, true);
  grid_map_pub_ = nh_.advertise<grid_map_msgs::GridMap>("grid_map", 1, true);
  current_pos_pub_ = nh_.advertise<geometry_msgs::PointStamped>("current_pos", 1, true);
  current_pose_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("current_pose", 1, true);

  // Load SVM model
  loadSVM(svm_path);
}

template <SamplingSpace SamplingSpaceType>
RmapPlanning<SamplingSpaceType>::~RmapPlanning()
{
  if (svm_mo_) {
    delete svm_mo_;
  }
}

template <SamplingSpace SamplingSpaceType>
void RmapPlanning<SamplingSpaceType>::configure(const mc_rtc::Configuration& mc_rtc_config)
{
  mc_rtc_config_ = mc_rtc_config;
  config_.load(mc_rtc_config);
}

template <SamplingSpace SamplingSpaceType>
void RmapPlanning<SamplingSpaceType>::setup()
{
  // Setup grid map
  setupGridMap();

  // Publish (MarkerArray message is fixed)
  publishMarkerArray();

  qp_coeff_.setup(vel_dim_, 0, 1);
  qp_coeff_.x_min_.setConstant(-config_.delta_config_limit);
  qp_coeff_.x_max_.setConstant(config_.delta_config_limit);

  qp_solver_ = OmgCore::allocateQpSolver(OmgCore::QpSolverType::JRLQP);

  current_sample_ = poseToSample<SamplingSpaceType>(sva::PTransformd::Identity());
}

template <SamplingSpace SamplingSpaceType>
void RmapPlanning<SamplingSpaceType>::runOnce()
{
  // Set QP coefficients
  qp_coeff_.obj_vec_ = sampleError<SamplingSpaceType>(target_sample_, current_sample_);
  double lambda = qp_coeff_.obj_vec_.squaredNorm() + 1e-3;
  qp_coeff_.obj_mat_.diagonal().setConstant(1.0 + lambda);
  setSVMIneq<SamplingSpaceType>(
      qp_coeff_.ineq_mat_.block(0, 0, 1, velDim<SamplingSpaceType>()),
      qp_coeff_.ineq_vec_.block(0, 0, 1, 1),
      current_sample_,
      svm_mo_->param,
      svm_mo_,
      svm_coeff_vec_,
      svm_sv_mat_,
      config_.svm_thre);

  // Solve QP
  const VelType& vel = qp_solver_->solve(qp_coeff_);

  // Integrate
  integrateVelToSample<SamplingSpaceType>(current_sample_, vel);

  // Publish
  publishCurrentState();

  // Predict SVM
  predictOnSlicePlane();
}

template <SamplingSpace SamplingSpaceType>
void RmapPlanning<SamplingSpaceType>::runLoop()
{
  setup();

  ros::Rate rate(100);
  while (ros::ok()) {
    runOnce();

    rate.sleep();
    ros::spinOnce();
  }
}

template <SamplingSpace SamplingSpaceType>
void RmapPlanning<SamplingSpaceType>::setupGridMap()
{
  // Calculate sample min/max
  // \todo Load from file
  SampleType sample_min = SampleType::Constant(2.0);
  SampleType sample_max = SampleType::Constant(-2.0);
  SampleType sample_range = sample_max - sample_min;
  {
    sample_min -= config_.grid_map_margin_ratio * sample_range;
    sample_max += config_.grid_map_margin_ratio * sample_range;
    sample_range = sample_max - sample_min;
  }

  // Create grid map
  {
    grid_map_ = std::make_shared<grid_map::GridMap>(std::vector<std::string>{"svm_value"});

    SampleType sample_center = (sample_min + sample_max) / 2;
    grid_map_->setFrameId("world");
    grid_map_->setGeometry(grid_map::Length(sample_range[0], sample_range[1]),
                           config_.grid_map_resolution,
                           grid_map::Position(sample_center[0], sample_center[1]));
  }
}

template <SamplingSpace SamplingSpaceType>
void RmapPlanning<SamplingSpaceType>::loadSVM(const std::string& svm_path)
{
  ROS_INFO_STREAM("Load SVM model from " << svm_path);
  svm_mo_ = svm_load_model(svm_path.c_str());

  setSVMPredictionMat();
}

template <SamplingSpace SamplingSpaceType>
void RmapPlanning<SamplingSpaceType>::setSVMPredictionMat()
{
  int num_sv = svm_mo_->l;
  svm_coeff_vec_.resize(num_sv);
  svm_sv_mat_.resize(input_dim_, num_sv);
  for (int i = 0; i < num_sv; i++) {
    svm_coeff_vec_[i] = svm_mo_->sv_coef[0][i];
    svm_sv_mat_.col(i) = svmNodeToEigenVec<SamplingSpaceType>(svm_mo_->SV[i]);
  }
}

template <SamplingSpace SamplingSpaceType>
void RmapPlanning<SamplingSpaceType>::predictOnSlicePlane()
{
  // Predict
  {
    auto start_time = std::chrono::system_clock::now();

    sva::PTransformd slice_origin = sampleToPose<SamplingSpaceType>(current_sample_);

    size_t grid_idx = 0;
    SampleType origin_sample = poseToSample<SamplingSpaceType>(slice_origin);
    for (grid_map::GridMapIterator it(*grid_map_); !it.isPastEnd(); ++it) {
      grid_map::Position pos;
      grid_map_->getPosition(*it, pos);

      SampleType sample = origin_sample;
      sample.x() = pos.x();
      if constexpr (sample_dim_ > 1) {
          sample.y() = pos.y();
        }

      // Calculate SVM value
      double svm_value = calcSVMValue<SamplingSpaceType>(
          sample,
          svm_mo_->param,
          svm_mo_,
          svm_coeff_vec_,
          svm_sv_mat_);
      grid_map_->at("svm_value", *it) = config_.grid_map_height_scale * svm_value;

      grid_idx++;
    }

    double duration = 1e3 * std::chrono::duration_cast<std::chrono::duration<double>>(
        std::chrono::system_clock::now() - start_time).count();
    ROS_INFO_STREAM("SVM predict duration: " << duration << " [ms] (predict-one: " <<
                    duration / grid_idx <<" [ms])");
  }

  // Publish
  {
    auto start_time = std::chrono::system_clock::now();

    grid_map_->setTimestamp(ros::Time::now().toNSec());
    grid_map_msgs::GridMap grid_map_msg;
    grid_map::GridMapRosConverter::toMessage(*grid_map_, grid_map_msg);
    grid_map_pub_.publish(grid_map_msg);

    double duration = 1e3 * std::chrono::duration_cast<std::chrono::duration<double>>(
        std::chrono::system_clock::now() - start_time).count();
    // Publish is fast compared with other process
    // ROS_INFO_STREAM("SVM publish duration: " << duration << " [ms]");
  }
}

template <SamplingSpace SamplingSpaceType>
void RmapPlanning<SamplingSpaceType>::publishMarkerArray() const
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

  // XY plane marker
  visualization_msgs::Marker xy_plane_marker;
  double plane_thickness = 0.01;
  xy_plane_marker.header = header_msg;
  xy_plane_marker.ns = "xy_plane";
  xy_plane_marker.id = marker_arr_msg.markers.size();
  xy_plane_marker.type = visualization_msgs::Marker::CUBE;
  xy_plane_marker.color = OmgCore::toColorRGBAMsg({0.8, 0.8, 0.8, 1.0});
  xy_plane_marker.scale.x = 100.0;
  xy_plane_marker.scale.y = 100.0;
  xy_plane_marker.scale.z = plane_thickness;
  xy_plane_marker.pose = OmgCore::toPoseMsg(
      sva::PTransformd(Eigen::Vector3d(0, 0, config_.svm_thre - 0.5 * plane_thickness)));
  marker_arr_msg.markers.push_back(xy_plane_marker);

  marker_arr_pub_.publish(marker_arr_msg);
}

template <SamplingSpace SamplingSpaceType>
void RmapPlanning<SamplingSpaceType>::publishCurrentState() const
{
  std_msgs::Header header_msg;
  header_msg.frame_id = "world";
  header_msg.stamp = ros::Time::now();

  // Publish point
  geometry_msgs::PointStamped pos_msg;
  pos_msg.header = header_msg;
  pos_msg.point = OmgCore::toPointMsg(sampleToCloudPos<SamplingSpaceType>(current_sample_));
  current_pos_pub_.publish(pos_msg);

  // Publish pose
  geometry_msgs::PoseStamped pose_msg;
  pose_msg.header = header_msg;
  pose_msg.pose = OmgCore::toPoseMsg(sampleToPose<SamplingSpaceType>(current_sample_));
  current_pose_pub_.publish(pose_msg);
}

template <SamplingSpace SamplingSpaceType>
void RmapPlanning<SamplingSpaceType>::transCallback(
    const geometry_msgs::TransformStamped::ConstPtr& trans_st_msg)
{
  if (trans_st_msg->child_frame_id == "target") {
    target_sample_ = poseToSample<SamplingSpaceType>(OmgCore::toSvaPTransform(trans_st_msg->transform));
  }
}

std::shared_ptr<RmapPlanningBase> DiffRmap::createRmapPlanning(
    SamplingSpace sampling_space,
    const std::string& svm_path)
{
  if (sampling_space == SamplingSpace::R2) {
    return std::make_shared<RmapPlanning<SamplingSpace::R2>>(svm_path);
  } else if (sampling_space == SamplingSpace::SO2) {
    return std::make_shared<RmapPlanning<SamplingSpace::SO2>>(svm_path);
  } else if (sampling_space == SamplingSpace::SE2) {
    return std::make_shared<RmapPlanning<SamplingSpace::SE2>>(svm_path);
  } else if (sampling_space == SamplingSpace::R3) {
    return std::make_shared<RmapPlanning<SamplingSpace::R3>>(svm_path);
  } else if (sampling_space == SamplingSpace::SO3) {
    return std::make_shared<RmapPlanning<SamplingSpace::SO3>>(svm_path);
  } else if (sampling_space == SamplingSpace::SE3) {
    return std::make_shared<RmapPlanning<SamplingSpace::SE3>>(svm_path);
  } else {
    mc_rtc::log::error_and_throw<std::runtime_error>(
        "[createRmapPlanning] Unsupported SamplingSpace: {}", std::to_string(sampling_space));
  }
}

// Declare template specialized class
// See https://stackoverflow.com/a/8752879
template class RmapPlanning<SamplingSpace::R2>;
template class RmapPlanning<SamplingSpace::SO2>;
template class RmapPlanning<SamplingSpace::SE2>;
template class RmapPlanning<SamplingSpace::R3>;
template class RmapPlanning<SamplingSpace::SO3>;
template class RmapPlanning<SamplingSpace::SE3>;
