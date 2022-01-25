/* Author: Masaki Murooka */

#pragma once

#include <mc_rtc/Configuration.h>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud.h>

#include <optmotiongen/Utils/RobotUtils.h>

#include <differentiable_rmap/SamplingUtils.h>


namespace DiffRmap
{
/** \brief Virtual base class to generate samples for reachability map. */
class RmapSamplingBase
{
 public:
  /** \brief Configure from mc_rtc configuration.
      \param mc_rtc_config mc_rtc configuration
   */
  virtual void configure(const mc_rtc::Configuration& mc_rtc_config) = 0;

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
  /*! \brief Configuration. */
  struct Configuration
  {
    //! Interval of loop counts to publish ROS message
    int publish_loop_interval = 100;

    //! Robot root pose
    sva::PTransformd root_pose = sva::PTransformd::Identity();

    //! Body pose offset
    sva::PTransformd body_pose_offset = sva::PTransformd::Identity();

    /*! \brief Load mc_rtc configuration. */
    inline virtual void load(const mc_rtc::Configuration& mc_rtc_config)
    {
      mc_rtc_config("publish_loop_interval", publish_loop_interval);
      mc_rtc_config("root_pose", root_pose);
      mc_rtc_config("body_pose_offset", body_pose_offset);
    }
  };

 public:
  /*! \brief Dimension of sample. */
  static constexpr int sample_dim_ = sampleDim<SamplingSpaceType>();

 public:
  /*! \brief Type of sample vector. */
  using SampleType = Sample<SamplingSpaceType>;

 public:
  /** \brief Constructor.
      \param rb robot
      \param body_name name of body whose pose is sampled
      \param joint_name_list name list of joints whose position is changed
  */
  RmapSampling(const std::shared_ptr<OmgCore::Robot>& rb,
               const std::string& body_name,
               const std::vector<std::string>& joint_name_list);

  /** \brief Configure from mc_rtc configuration.
      \param mc_rtc_config mc_rtc configuration
   */
  virtual void configure(const mc_rtc::Configuration& mc_rtc_config) override;

  /** \brief Run sample generation
      \param bag_path path of ROS bag file
      \param sample_num number of samples to be generated
      \param sleep_rate rate of sleep druing sample generation. zero for no sleep
  */
  virtual void run(const std::string& bag_path = "/tmp/rmap_sample_set.bag",
                   int sample_num = 10000,
                   double sleep_rate = 0) override;

  /** \brief Accessor to Robot array. */
  inline const OmgCore::RobotArray& rbArr() const
  {
    return rb_arr_;
  }

 protected:
  /** \brief Constructor.
      \param rb robot
  */
  RmapSampling(const std::shared_ptr<OmgCore::Robot>& rb);

  /** \brief Setup sampling. */
  virtual void setupSampling();

  /** \brief Generate one sample. */
  virtual void sampleOnce(int sample_idx);

  /** \brief Publish ROS message. */
  virtual void publish();

  /** \brief Dump generated sample set to ROS bag. */
  void dumpSampleSet(const std::string& bag_path) const;

 protected:
  //! mc_rtc Configuration
  mc_rtc::Configuration mc_rtc_config_;

  //! Configuration
  Configuration config_;

  //! Robot array (single robot is stored)
  OmgCore::RobotArray rb_arr_;
  //! Robot configuration array (single robot configuration is stored)
  OmgCore::RobotConfigArray rbc_arr_;

  //! Name of body whose pose is sampled
  std::string body_name_;
  //! Index of body whose pose is sampled
  int body_idx_;

  //! Name list of joints whose position is changed
  std::vector<std::string> joint_name_list_;
  //! Index list of joints whose position is changed
  std::vector<int> joint_idx_list_;
  //! Joint position coefficient to make sample from [-1:1] random value
  Eigen::VectorXd joint_pos_coeff_;
  //! Joint position offset to make sample from [-1:1] random value
  Eigen::VectorXd joint_pos_offset_;

  //! Sample list
  std::vector<SampleType> sample_list_;

  //! Reachability list
  std::vector<bool> reachability_list_;

  //! ROS related members
  ros::NodeHandle nh_;

  ros::Publisher rs_arr_pub_;
  ros::Publisher reachable_cloud_pub_;
  ros::Publisher unreachable_cloud_pub_;

  sensor_msgs::PointCloud reachable_cloud_msg_;
  sensor_msgs::PointCloud unreachable_cloud_msg_;
};

/** \brief Create RmapSampling instance.
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
