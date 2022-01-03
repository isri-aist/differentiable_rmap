/* Author: Masaki Murooka */

#pragma once

#include <ros/ros.h>

#include <optmotiongen/Utils/RobotUtils.h>

#include <differentiable_rmap/SamplingUtils.h>


namespace DiffRmap
{
/** \brief Virtual base class to generate samples for reachability map. */
class RmapSamplingBase
{
 public:
  /** \brief Run sample generation
      \param bag_path path of ROS bag file
      \param sample_num number of samples to be generated
      \param sleep_rate rate of sleep druing sample generation. zero for no sleep
  */
  virtual void run(const std::string& bag_path,
                   int sample_num,
                   double sleep_rate) = 0;
};

/** \brief Class to generate samples for reachability map.
    \tparam SamplingSpaceType sampling space
 */
template <SamplingSpace SamplingSpaceType>
class RmapSampling: public RmapSamplingBase
{
 public:
  /*! \brief Dimension of sample. */
  static constexpr int SampleDim = sampleDim<SamplingSpaceType>();

 public:
  /*! \brief Type of sample vector. */
  using SampleVector = Eigen::Matrix<double, SampleDim, 1>;

 public:
  /** \brief Constructor.
      \param rb robot
      \param body_name name of body whose pose is sampled
      \param joint_name_list name list of joints whose position is changed
  */
  RmapSampling(const std::shared_ptr<OmgCore::Robot>& rb,
               const std::string& body_name,
               const std::vector<std::string>& joint_name_list);

  /** \brief Run sample generation
      \param bag_path path of ROS bag file
      \param sample_num number of samples to be generated
      \param sleep_rate rate of sleep druing sample generation. zero for no sleep
  */
  virtual void run(const std::string& bag_path = "/tmp/rmap_sample_set.bag",
                   int sample_num = 10000,
                   double sleep_rate = 0) override;

 protected:
  /** \brief Dump generated sample set to ROS bag. */
  void dumpBag(const std::string& bag_path) const;

 public:
  OmgCore::RobotArray rb_arr_;
  OmgCore::RobotConfigArray rbc_arr_;

  std::string body_name_;
  int body_idx_;

  std::vector<std::string> joint_name_list_;
  std::vector<int> joint_idx_list_;
  Eigen::VectorXd joint_pos_coeff_;
  Eigen::VectorXd joint_pos_offset_;

  std::vector<SampleVector> sample_list_;

  ros::NodeHandle nh_;

  ros::Publisher rs_arr_pub_;
  ros::Publisher rmap_cloud_pub_;
};

/** \brief Constructor.
    \param sampling_space sampling space
    \param rb robot
    \param body_name name of body whose pose is sampled
    \param joint_name_list name list of joints whose position is changed
*/
std::shared_ptr<RmapSamplingBase> createRmapSampling(
    SamplingSpace sampling_space,
    const std::shared_ptr<OmgCore::Robot>& rb,
    const std::string& body_name,
    const std::vector<std::string>& joint_name_list);
}
