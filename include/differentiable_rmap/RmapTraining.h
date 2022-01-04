/* Author: Masaki Murooka */

#pragma once

#include <mc_rtc/Configuration.h>

#include <ros/ros.h>
#include <grid_map_ros/grid_map_ros.hpp>
#include <grid_map_msgs/GridMap.h>

#include <libsvm/svm.h>

#include <differentiable_rmap/SamplingUtils.h>


namespace DiffRmap
{
/** \brief Virtual base class to train SVM for differentiable reachability map. */
class RmapTrainingBase
{
 public:
  /** \brief Run SVM training. */
  virtual void run() = 0;
};

/** \brief Class to train SVM for differentiable reachability map.
    \tparam SamplingSpaceType sampling space
 */
template <SamplingSpace SamplingSpaceType>
class RmapTraining: public RmapTrainingBase
{
 public:
  /*! \brief Configuration. */
  struct Configuration
  {
    double grid_map_margin_ratio = 0.5;

    double grid_map_resolution = 0.02;

    double grid_map_height_scale = 1.0;
  };

 public:
  /*! \brief Dimension of sample. */
  static constexpr int sample_dim_ = sampleDim<SamplingSpaceType>();

  /*! \brief Dimension of SVM input. */
  static constexpr int input_dim_ = inputDim<SamplingSpaceType>();

 public:
  /*! \brief Type of sample vector. */
  using SampleVector = Eigen::Matrix<double, sample_dim_, 1>;

  /*! \brief Type of input vector. */
  using InputVector = Eigen::Matrix<double, input_dim_, 1>;

 public:
  /** \brief Constructor.
      \param bag_path path of ROS bag file (empty for loading trained SVM model directly)
      \param svm_path path of SVM model file
   */
  RmapTraining(const std::string& bag_path = "/tmp/rmap_sample_set.bag",
               const std::string& svm_path = "/tmp/rmap_svm_model.libsvm");

  /** \brief Destructor. */
  ~RmapTraining();

  /** \brief Run SVM training. */
  virtual void run() override;

 protected:
  /** \brief Setup SVM parameter. */
  void setupSVMParam();

  /** \brief Setup grid map. */
  void setupGridMap();

  /** \brief Train SVM. */
  void train();

  /** \brief Predict SVM. */
  void predict();

  /** \brief Load sample set from ROS bag. */
  void loadBag(const std::string& bag_path);

  /** \brief Save SVM model. */
  void loadSVM();

 public:
  Configuration config_;

  std::vector<SampleVector> sample_list_;

  bool svm_loaded_;
  std::string svm_path_;

  svm_node input_node_[input_dim_ + 1];
  svm_node* all_input_nodes_;
  svm_problem svm_prob_;
  svm_parameter svm_param_;
  svm_model *svm_mo_;

  std::shared_ptr<grid_map::GridMap> grid_map_;

  ros::NodeHandle nh_;

  ros::Publisher rmap_cloud_pub_;
  ros::Publisher grid_map_pub_;
};

/** \brief Constructor.
    \param sampling_space sampling space
*/
std::shared_ptr<RmapTrainingBase> createRmapTraining(
    SamplingSpace sampling_space,
    const std::string& bag_path = "/tmp/rmap_sample_set.bag",
    const std::string& svm_path = "/tmp/rmap_svm_model.libsvm");
}
