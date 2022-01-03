/* Author: Masaki Murooka */

#include <mc_rtc/logging.h>

#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/PointCloud.h>
#include <optmotiongen_msgs/RobotStateArray.h>
#include <differentiable_rmap/RmapSampleSet.h>

#include <optmotiongen/Utils/RosUtils.h>

#include <differentiable_rmap/RmapSampling.h>

using namespace DiffRmap;


template <SamplingSpace SamplingSpaceType>
RmapSampling<SamplingSpaceType>::RmapSampling(
    const std::shared_ptr<OmgCore::Robot>& rb,
    const std::string& body_name,
    const std::vector<std::string>& joint_name_list):
    body_name_(body_name),
    body_idx_(rb->bodyIndexByName(body_name_)),
    joint_name_list_(joint_name_list)
{
  // Setup robot
  rb_arr_.push_back(rb);
  rb_arr_.setup();
  rbc_arr_ = OmgCore::RobotConfigArray(rb_arr_);

  // Setup callback
  rs_arr_pub_ = nh_.advertise<optmotiongen_msgs::RobotStateArray>("robot_state_arr", 1, true);
  rmap_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud>("rmap_cloud", 1, true);

  // Setup sampling
  joint_idx_list_.resize(joint_name_list_.size());
  joint_pos_coeff_.resize(joint_name_list_.size());
  joint_pos_offset_.resize(joint_name_list_.size());
  {
    for (size_t i = 0; i < joint_name_list_.size(); i++) {
      const auto& joint_name = joint_name_list_[i];
      joint_idx_list_[i] = rb_arr_[0]->jointIndexByName(joint_name);
      double lower_joint_pos = rb_arr_[0]->limits_.lower.at(joint_name)[0];
      double upper_joint_pos = rb_arr_[0]->limits_.upper.at(joint_name)[0];
      joint_pos_coeff_[i] = (upper_joint_pos - lower_joint_pos) / 2;
      joint_pos_offset_[i] = (upper_joint_pos + lower_joint_pos) / 2;
    }
  }
}

template <SamplingSpace SamplingSpaceType>
void RmapSampling<SamplingSpaceType>::run(
    const std::string& bag_path,
    int sample_num,
    double sleep_rate)
{
  const auto& rb = rb_arr_[0];
  const auto& rbc = rbc_arr_[0];

  sensor_msgs::PointCloud cloud_msg;
  cloud_msg.header.frame_id = "world";

  ros::Rate rate(sleep_rate > 0 ? sleep_rate : 1000);
  int loop_idx = 0;
  while (ros::ok()) {
    if (loop_idx == sample_num) {
      break;
    }

    // Sample random posture
    {
      Eigen::VectorXd joint_pos =
          joint_pos_coeff_.cwiseProduct(Eigen::VectorXd::Random(joint_name_list_.size())) + joint_pos_offset_;
      for (size_t i = 0; i < joint_name_list_.size(); i++) {
        rbc->q[joint_idx_list_[i]][0] = joint_pos[i];
      }
      rbd::forwardKinematics(*rb, *rbc);
      const auto& body_pose = rbc->bodyPosW[body_idx_];
      const SampleVector& sample = poseToSample<SamplingSpaceType>(body_pose);
      sample_list_.push_back(sample);
      Eigen::Vector3d cloud_pos = body_pose.translation();
      if constexpr (SamplingSpaceType == SamplingSpace::R2) {
        cloud_pos.z() = 0;
      } else if constexpr (SamplingSpaceType == SamplingSpace::SE2) {
        cloud_pos.z() = sample.z();
      }
      cloud_msg.points.push_back(OmgCore::toPoint32Msg(cloud_pos));
    }

    if(loop_idx % 100 == 0) {
      // Publish robot
      rs_arr_pub_.publish(rb_arr_.makeRobotStateArrayMsg(rbc_arr_));

      // Publish cloud
      cloud_msg.header.stamp = ros::Time::now();
      rmap_cloud_pub_.publish(cloud_msg);
    }

    if (sleep_rate > 0) {
      rate.sleep();
    }
    ros::spinOnce();
    loop_idx++;
  }

  dumpBag(bag_path);
}

template <SamplingSpace SamplingSpaceType>
void RmapSampling<SamplingSpaceType>::dumpBag(const std::string& bag_path) const
{
  rosbag::Bag bag(bag_path, rosbag::bagmode::Write);

  differentiable_rmap::RmapSampleSet sample_set_msg;
  sample_set_msg.type = static_cast<size_t>(SamplingSpaceType);
  sample_set_msg.samples.resize(sample_list_.size());
  for (size_t i = 0; i < sample_list_.size(); i++) {
    const auto& sample = sample_list_[i];
    sample_set_msg.samples[i].position.resize(sample.size());
    for (int j = 0; j < sample.size(); j++) {
      sample_set_msg.samples[i].position[j] = sample[j];
    }
    sample_set_msg.samples[i].is_reachable = true;
  }
  bag.write("/rmap_sample_set", ros::Time::now(), sample_set_msg);
  ROS_INFO_STREAM("Dump sample set to " << bag_path);
}

std::shared_ptr<RmapSamplingBase> DiffRmap::createRmapSampling(
    SamplingSpace sampling_space,
    const std::shared_ptr<OmgCore::Robot>& rb,
    const std::string& body_name,
    const std::vector<std::string>& joint_name_list)
{
  if (sampling_space == SamplingSpace::R2) {
    return std::make_shared<RmapSampling<SamplingSpace::R2>>(rb, body_name, joint_name_list);
  } else if (sampling_space == SamplingSpace::SO2) {
    return std::make_shared<RmapSampling<SamplingSpace::SO2>>(rb, body_name, joint_name_list);
  } else if (sampling_space == SamplingSpace::SE2) {
    return std::make_shared<RmapSampling<SamplingSpace::SE2>>(rb, body_name, joint_name_list);
  } else if (sampling_space == SamplingSpace::R3) {
    return std::make_shared<RmapSampling<SamplingSpace::R3>>(rb, body_name, joint_name_list);
  } else if (sampling_space == SamplingSpace::SO3) {
    return std::make_shared<RmapSampling<SamplingSpace::SO3>>(rb, body_name, joint_name_list);
  } else if (sampling_space == SamplingSpace::SE3) {
    return std::make_shared<RmapSampling<SamplingSpace::SE3>>(rb, body_name, joint_name_list);
  } else {
    mc_rtc::log::error_and_throw<std::runtime_error>(
        "[strToSamplingSpace] Unsupported SamplingSpace name: {}", std::to_string(sampling_space));
  }
}
