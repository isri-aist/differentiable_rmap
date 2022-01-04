/* Author: Masaki Murooka */

#pragma once

#include <ros/ros.h>

#include <differentiable_rmap/SamplingUtils.h>


namespace DiffRmap
{
/** \brief Virtual base class to train SVM for differentiable reachability map. */
class RmapTrainingBase
{
 public:
  /** \brief Run SVM training.
      \param bag_path path of ROS bag file
  */
  virtual void run(const std::string& bag_path) = 0;
};

/** \brief Class to train SVM for differentiable reachability map.
    \tparam SamplingSpaceType sampling space
 */
template <SamplingSpace SamplingSpaceType>
class RmapTraining
{
 public:
  /*! \brief Dimension of sample. */
  static constexpr int sample_dim_ = sampleDim<SamplingSpaceType>();

 public:
  /*! \brief Type of sample vector. */
  using SampleVector = Eigen::Matrix<double, sample_dim_, 1>;

 public:
  /** \brief Constructor. */
  RmapTraining();

  /** \brief Run SVM training.
      \param bag_path path of ROS bag file
  */
  void run(const std::string& bag_path = "/tmp/rmap_sample_set.bag");

 protected:
  /** \brief Load sample set from ROS bag. */
  void loadBag(const std::string& bag_path);

 public:
  std::vector<SampleVector> sample_list_;

  ros::NodeHandle nh_;

  ros::Publisher rmap_cloud_pub_;
};
}
