/* Author: Masaki Murooka */

#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/PointCloud.h>
#include <differentiable_rmap/RmapSampleSet.h>

#include <optmotiongen/Utils/RosUtils.h>

#include <differentiable_rmap/RmapTraining.h>

using namespace DiffRmap;


template <SamplingSpace SamplingSpaceType>
RmapTraining<SamplingSpaceType>::RmapTraining()
{
  // Setup callback
  rmap_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud>("rmap_cloud", 1, true);
}

template <SamplingSpace SamplingSpaceType>
void RmapTraining<SamplingSpaceType>::run(const std::string& bag_path)
{
  loadBag(bag_path);

  // Publish cloud
  {
    std_msgs::Header header_msg;
    header_msg.frame_id = "world";
    header_msg.stamp = ros::Time::now();

    sensor_msgs::PointCloud cloud_msg;
    cloud_msg.header = header_msg;
    cloud_msg.points.resize(sample_list_.size());
    for (size_t i = 0; i < sample_list_.size(); i++) {
      cloud_msg.points[i] = OmgCore::toPoint32Msg(sampleToCloudPos<SamplingSpaceType>(sample_list_[i]));
    }
    rmap_cloud_pub_.publish(cloud_msg);
  }
}

template <SamplingSpace SamplingSpaceType>
void RmapTraining<SamplingSpaceType>::loadBag(const std::string& bag_path)
{
  ROS_INFO_STREAM("Load sample set from " << bag_path);
  rosbag::Bag bag(bag_path, rosbag::bagmode::Read);
  int cnt = 0;
  for (rosbag::MessageInstance const m :
           rosbag::View(bag, rosbag::TopicQuery(std::vector<std::string>{"/rmap_sample_set"}))) {
    if (cnt >= 1) {
      ROS_WARN("Multiple messages are stored in bag file. load only first one.");
      break;
    }

    differentiable_rmap::RmapSampleSet::ConstPtr sample_set_msg =
        m.instantiate<differentiable_rmap::RmapSampleSet>();
    if (sample_set_msg == nullptr) {
      mc_rtc::log::error_and_throw<std::runtime_error>("Failed to load sample set message from rosbag.");
    }
    if (sample_set_msg->type != static_cast<size_t>(SamplingSpaceType)) {
      mc_rtc::log::error_and_throw<std::runtime_error>(
          "SamplingSpace does not match with message: {} != {}",
          sample_set_msg->type, static_cast<size_t>(SamplingSpaceType));
    }
    sample_list_.resize(sample_set_msg->samples.size());
    for (size_t i = 0; i < sample_set_msg->samples.size(); i++) {
      for (int j = 0; j < sample_dim_; j++) {
        sample_list_[i][j] = sample_set_msg->samples[i].position[j];
      }
    }

    cnt++;
  }
  if (cnt == 0) {
    mc_rtc::log::error_and_throw<std::runtime_error>("Sample set message not found.");
  }
}

std::shared_ptr<RmapTrainingBase> DiffRmap::createRmapTraining(
    SamplingSpace sampling_space)
{
  if (sampling_space == SamplingSpace::R2) {
    return std::make_shared<RmapTraining<SamplingSpace::R2>>();
  } else if (sampling_space == SamplingSpace::SO2) {
    return std::make_shared<RmapTraining<SamplingSpace::SO2>>();
  } else if (sampling_space == SamplingSpace::SE2) {
    return std::make_shared<RmapTraining<SamplingSpace::SE2>>();
  } else if (sampling_space == SamplingSpace::R3) {
    return std::make_shared<RmapTraining<SamplingSpace::R3>>();
  } else if (sampling_space == SamplingSpace::SO3) {
    return std::make_shared<RmapTraining<SamplingSpace::SO3>>();
  } else if (sampling_space == SamplingSpace::SE3) {
    return std::make_shared<RmapTraining<SamplingSpace::SE3>>();
  } else {
    mc_rtc::log::error_and_throw<std::runtime_error>(
        "[createRmapTraining] Unsupported SamplingSpace: {}", std::to_string(sampling_space));
  }
}
