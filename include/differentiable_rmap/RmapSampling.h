/* Author: Masaki Murooka */

#pragma once

#include <ros/ros.h>

#include <optmotiongen/Utils/RobotUtils.h>

#include <differentiable_rmap/SamplingUtils.h>


namespace DiffRmap
{
/** \brief Class to generate samples for reachability map.
*/
class RmapSampling
{
 public:
  /** \brief Constructor.
      \param rb robot
      \param sampling_space sampling space
      \param body_name name of body whose pose is sampled
      \param joint_name_list name list of joints whose position is changed
  */
  RmapSampling(const std::shared_ptr<OmgCore::Robot>& rb,
               SamplingSpace sampling_space,
               const std::string& body_name,
               const std::vector<std::string>& joint_name_list);

  /** \brief Run sample generation
      \param bag_path path of ROS bag file
      \param sample_num number of samples to be generated
      \param sleep_rate rate of sleep druing sample generation. zero for no sleep
  */
  void run(const std::string& bag_path = "/tmp/rmap_sample_set.bag",
           int sample_num = 10000,
           double sleep_rate = 0);

 public:
  OmgCore::RobotArray rb_arr_;
  OmgCore::RobotConfigArray rbc_arr_;

  SamplingSpace sampling_space_;

  std::string body_name_;
  int body_idx_;

  std::vector<std::string> joint_name_list_;
  std::vector<int> joint_idx_list_;
  Eigen::VectorXd joint_pos_coeff_;
  Eigen::VectorXd joint_pos_offset_;

  std::vector<Eigen::VectorXd> sample_list_;

  ros::NodeHandle nh_;

  ros::Publisher rs_arr_pub_;
  ros::Publisher rmap_cloud_pub_;
};
}
