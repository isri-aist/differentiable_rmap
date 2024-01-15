/* Author: Masaki Murooka */

#include <chrono>
#include <stdlib.h>

#include <optmotiongen_msgs/RobotStateArray.h>
#include <visualization_msgs/MarkerArray.h>
#include <differentiable_rmap/RmapSampleSet.h>
#include <rosbag/bag.h>

#include <sch/S_Polyhedron/S_Polyhedron.h>

#include <optmotiongen_core/Utils/RosUtils.h>

#include <differentiable_rmap/RmapSampling.h>

using namespace DiffRmap;

template<SamplingSpace SamplingSpaceType>
RmapSampling<SamplingSpaceType>::RmapSampling(const std::shared_ptr<OmgCore::Robot> & rb)
{
  // Setup robot
  rb_arr_.push_back(rb);
  rb_arr_.setup();
  rbc_arr_ = OmgCore::RobotConfigArray(rb_arr_);

  // Setup ROS
  rs_arr_pub_ = nh_.advertise<optmotiongen_msgs::RobotStateArray>("robot_state_arr", 1, true);
  reachable_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud>("reachable_cloud", 1, true);
  unreachable_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud>("unreachable_cloud", 1, true);
  collision_marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("collision_marker", 1, true);
}

template<SamplingSpace SamplingSpaceType>
RmapSampling<SamplingSpaceType>::RmapSampling(const std::shared_ptr<OmgCore::Robot> & rb,
                                              const std::string & body_name,
                                              const std::vector<std::string> & joint_name_list)
: RmapSampling(rb)
{
  // Setup body and joint
  body_name_ = body_name;
  body_idx_ = rb->bodyIndexByName(body_name_);
  joint_name_list_ = joint_name_list;
}

template<SamplingSpace SamplingSpaceType>
void RmapSampling<SamplingSpaceType>::configure(const mc_rtc::Configuration & mc_rtc_config)
{
  mc_rtc_config_ = mc_rtc_config;
  config_.load(mc_rtc_config);
}

template<SamplingSpace SamplingSpaceType>
void RmapSampling<SamplingSpaceType>::run(const std::string & bag_path, int sample_num, double sleep_rate)
{
  setup();

  sample_list_.resize(sample_num);
  reachability_list_.resize(sample_num);
  reachable_cloud_msg_.points.clear();
  unreachable_cloud_msg_.points.clear();

  auto start_time = std::chrono::system_clock::now();

  ros::Rate rate(sleep_rate > 0 ? sleep_rate : 1000);
  int loop_idx = 0;
  while(ros::ok())
  {
    if(loop_idx == sample_num)
    {
      break;
    }

    // Sample once
    while(!sampleOnce(loop_idx))
      ;

    if(loop_idx % config_.publish_loop_interval == 0)
    {
      publish();
    }

    if(sleep_rate > 0)
    {
      rate.sleep();
    }
    ros::spinOnce();
    loop_idx++;
  }

  double duration =
      1e3
      * std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::system_clock::now() - start_time).count();
  ROS_INFO_STREAM("Sample generation duration: " << duration << " [ms]");

  // Dump sample set
  dumpSampleSet(bag_path);
}

template<SamplingSpace SamplingSpaceType>
void RmapSampling<SamplingSpaceType>::setup()
{
  setupSampling();
  setupCollisionTask();
}

template<SamplingSpace SamplingSpaceType>
void RmapSampling<SamplingSpaceType>::setupSampling()
{
  // Eigen's random seed can be set by srand
  srand(config_.random_seed);

  // Set robot root pose
  rb_arr_[0]->rootPose(config_.root_pose);

  // Calculate coefficient and offset to make random position
  joint_idx_list_.resize(joint_name_list_.size());
  joint_pos_coeff_.resize(joint_name_list_.size());
  joint_pos_offset_.resize(joint_name_list_.size());
  {
    for(size_t i = 0; i < joint_name_list_.size(); i++)
    {
      const auto & joint_name = joint_name_list_[i];
      joint_idx_list_[i] = rb_arr_[0]->jointIndexByName(joint_name);
      double lower_joint_pos = rb_arr_[0]->limits_.lower.at(joint_name)[0];
      double upper_joint_pos = rb_arr_[0]->limits_.upper.at(joint_name)[0];
      joint_pos_coeff_[i] = (upper_joint_pos - lower_joint_pos) / 2;
      joint_pos_offset_[i] = (upper_joint_pos + lower_joint_pos) / 2;
    }
  }
}

template<SamplingSpace SamplingSpaceType>
void RmapSampling<SamplingSpaceType>::setupCollisionTask()
{
  // Since robot_convex_path needs to resolve the ROS package path, it is obtained by rosparam instead of mc_rtc
  // configuration
  std::string robot_convex_path;
  nh_.getParam("robot_convex_path", robot_convex_path);

  collision_task_list_.clear();
  for(const auto & body_names : config_.collision_body_names_list)
  {
    OmgCore::Twin<std::shared_ptr<sch::S_Object>> sch_objs;
    for(auto i : {0, 1})
    {
      sch_objs[i] = OmgCore::loadSchPolyhedron(robot_convex_path + body_names[i] + "_mesh-ch.txt");
    }
    auto task = std::make_shared<OmgCore::CollisionTask>(
        std::make_shared<OmgCore::CollisionFunc>(rb_arr_, OmgCore::Twin<int>{0, 0}, body_names, sch_objs), 0.05);
    task->setWeight(config_.collision_task_weight);
    collision_task_list_.push_back(task);
  }
}

template<SamplingSpace SamplingSpaceType>
bool RmapSampling<SamplingSpaceType>::sampleOnce(int sample_idx)
{
  const auto & rb = rb_arr_[0];
  const auto & rbc = rbc_arr_[0];

  // Set random configuration
  Eigen::VectorXd joint_pos =
      joint_pos_coeff_.cwiseProduct(Eigen::VectorXd::Random(joint_name_list_.size())) + joint_pos_offset_;
  for(size_t i = 0; i < joint_name_list_.size(); i++)
  {
    rbc->q[joint_idx_list_[i]][0] = joint_pos[i];
  }
  rbd::forwardKinematics(*rb, *rbc);

  // Check collision task
  bool has_collision = false;
  for(const auto & task : collision_task_list_)
  {
    task->update(rb_arr_, rbc_arr_, aux_rb_arr_);
    if(task->value().cwiseMax(0).squaredNorm() > 1e-6)
    {
      has_collision = true;
      break;
    }
  }

  // Append new sample to sample list
  if(!has_collision)
  {
    const auto & body_pose = config_.body_pose_offset * rbc->bodyPosW[body_idx_];
    const SampleType & sample = poseToSample<SamplingSpaceType>(body_pose);
    sample_list_[sample_idx] = sample;
    reachability_list_[sample_idx] = true;
    reachable_cloud_msg_.points.push_back(OmgCore::toPoint32Msg(sampleToCloudPos<SamplingSpaceType>(sample)));
  }

  return !has_collision;
}

template<SamplingSpace SamplingSpaceType>
void RmapSampling<SamplingSpaceType>::publish()
{
  // Publish robot
  rs_arr_pub_.publish(rb_arr_.makeRobotStateArrayMsg(rbc_arr_));

  // Publish cloud
  const auto & time_now = ros::Time::now();
  reachable_cloud_msg_.header.frame_id = "world";
  reachable_cloud_msg_.header.stamp = time_now;
  reachable_cloud_pub_.publish(reachable_cloud_msg_);
  unreachable_cloud_msg_.header.frame_id = "world";
  unreachable_cloud_msg_.header.stamp = time_now;
  unreachable_cloud_pub_.publish(unreachable_cloud_msg_);

  // Publish collision marker
  publishCollisionMarker(collision_task_list_);
}

template<SamplingSpace SamplingSpaceType>
void RmapSampling<SamplingSpaceType>::publishCollisionMarker(
    const std::vector<std::shared_ptr<OmgCore::CollisionTask>> & collision_task_list)
{
  visualization_msgs::MarkerArray marker_arr_msg;

  // delete marker
  visualization_msgs::Marker del_marker;
  del_marker.action = visualization_msgs::Marker::DELETEALL;
  del_marker.header.frame_id = "world";
  del_marker.id = marker_arr_msg.markers.size();
  marker_arr_msg.markers.push_back(del_marker);

  // point and line list marker connecting the closest points
  visualization_msgs::Marker closest_points_marker;
  closest_points_marker.header.frame_id = "world";
  closest_points_marker.ns = "closest_points";
  closest_points_marker.id = marker_arr_msg.markers.size();
  closest_points_marker.type = visualization_msgs::Marker::SPHERE_LIST;
  closest_points_marker.color = OmgCore::toColorRGBAMsg({0, 0, 1, 1});
  closest_points_marker.scale = OmgCore::toVector3Msg({0.02, 0.02, 0.02}); // sphere size
  closest_points_marker.pose.orientation = OmgCore::toQuaternionMsg({0, 0, 0, 1});
  visualization_msgs::Marker closest_lines_marker;
  closest_lines_marker.header.frame_id = "world";
  closest_lines_marker.ns = "closest_lines";
  closest_lines_marker.id = marker_arr_msg.markers.size();
  closest_lines_marker.type = visualization_msgs::Marker::LINE_LIST;
  closest_lines_marker.color = OmgCore::toColorRGBAMsg({0, 0, 1, 1});
  closest_lines_marker.scale.x = 0.01; // line width
  closest_lines_marker.pose.orientation = OmgCore::toQuaternionMsg({0, 0, 0, 1});
  for(const auto & collision_task : collision_task_list)
  {
    for(auto i : {0, 1})
    {
      closest_points_marker.points.push_back(OmgCore::toPointMsg(collision_task->func()->closest_points_[i]));
      closest_lines_marker.points.push_back(OmgCore::toPointMsg(collision_task->func()->closest_points_[i]));
    }
  }
  marker_arr_msg.markers.push_back(closest_points_marker);
  marker_arr_msg.markers.push_back(closest_lines_marker);

  collision_marker_pub_.publish(marker_arr_msg);
}

template<SamplingSpace SamplingSpaceType>
void RmapSampling<SamplingSpaceType>::dumpSampleSet(const std::string & bag_path) const
{
  differentiable_rmap::RmapSampleSet sample_set_msg;
  sample_set_msg.type = static_cast<size_t>(SamplingSpaceType);
  sample_set_msg.samples.resize(sample_list_.size());

  SampleType sample_min = SampleType::Constant(1e10);
  SampleType sample_max = SampleType::Constant(-1e10);

  // Since libsvm considers the first class to be positive,
  // add the reachable sample from the beginning and the unreachable sample from the end.
  size_t reachable_idx = 0;
  size_t unreachable_idx = 0;
  for(size_t i = 0; i < sample_list_.size(); i++)
  {
    const SampleType & sample = sample_list_[i];

    // Get msg_idx according to sample reachability
    size_t msg_idx;
    if(reachability_list_[i])
    {
      msg_idx = reachable_idx;
      reachable_idx++;
    }
    else
    {
      msg_idx = sample_list_.size() - 1 - unreachable_idx;
      unreachable_idx++;
    }

    // Set sample to message
    sample_set_msg.samples[msg_idx].position.resize(sample_dim_);
    for(int j = 0; j < sample_dim_; j++)
    {
      sample_set_msg.samples[msg_idx].position[j] = sample[j];
    }
    sample_set_msg.samples[msg_idx].is_reachable = reachability_list_[i];

    // Update min/max samples
    sample_min = sample_min.cwiseMin(sample);
    sample_max = sample_max.cwiseMax(sample);
  }

  // Set min/max samples to message
  sample_set_msg.min.resize(sample_dim_);
  sample_set_msg.max.resize(sample_dim_);
  for(int i = 0; i < sample_dim_; i++)
  {
    sample_set_msg.min[i] = sample_min[i];
    sample_set_msg.max[i] = sample_max[i];
  }

  // Dump to ROS bag
  rosbag::Bag bag(bag_path, rosbag::bagmode::Write);
  bag.write("/rmap_sample_set", ros::Time::now(), sample_set_msg);
  ROS_INFO_STREAM("Dump sample set to " << bag_path);
}

std::shared_ptr<RmapSamplingBase> DiffRmap::createRmapSampling(SamplingSpace sampling_space,
                                                               const std::shared_ptr<OmgCore::Robot> & rb,
                                                               const std::string & body_name,
                                                               const std::vector<std::string> & joint_name_list)
{
  if(sampling_space == SamplingSpace::R2)
  {
    return std::make_shared<RmapSampling<SamplingSpace::R2>>(rb, body_name, joint_name_list);
  }
  else if(sampling_space == SamplingSpace::SO2)
  {
    return std::make_shared<RmapSampling<SamplingSpace::SO2>>(rb, body_name, joint_name_list);
  }
  else if(sampling_space == SamplingSpace::SE2)
  {
    return std::make_shared<RmapSampling<SamplingSpace::SE2>>(rb, body_name, joint_name_list);
  }
  else if(sampling_space == SamplingSpace::R3)
  {
    return std::make_shared<RmapSampling<SamplingSpace::R3>>(rb, body_name, joint_name_list);
  }
  else if(sampling_space == SamplingSpace::SO3)
  {
    return std::make_shared<RmapSampling<SamplingSpace::SO3>>(rb, body_name, joint_name_list);
  }
  else if(sampling_space == SamplingSpace::SE3)
  {
    return std::make_shared<RmapSampling<SamplingSpace::SE3>>(rb, body_name, joint_name_list);
  }
  else
  {
    mc_rtc::log::error_and_throw<std::runtime_error>("[createRmapSampling] Unsupported SamplingSpace: {}",
                                                     std::to_string(sampling_space));
  }
}

// Declare template specialized class
// See https://stackoverflow.com/a/8752879
template class RmapSampling<SamplingSpace::R2>;
template class RmapSampling<SamplingSpace::SO2>;
template class RmapSampling<SamplingSpace::SE2>;
template class RmapSampling<SamplingSpace::R3>;
template class RmapSampling<SamplingSpace::SO3>;
template class RmapSampling<SamplingSpace::SE3>;
