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
  /*! \brief Configuration. */
  struct Configuration
  {
    //! Margin ratio of grid map
    double grid_map_margin_ratio = 0.5;

    //! Resolution of grid map
    double grid_map_resolution = 0.02;

    //! Height scale of grid map
    double grid_map_height_scale = 1.0;

    //! Height of xy plane marker
    double xy_plane_height = 0.0;
  };

 public:
  /** \brief Run SVM training. */
  virtual void run() = 0;

  /** \brief Configure from YAML file. */
  void configure(const std::string& config_path);

 protected:
  //! Configuration
  Configuration config_;
};

/** \brief Class to train SVM for differentiable reachability map.
    \tparam SamplingSpaceType sampling space
 */
template <SamplingSpace SamplingSpaceType>
class RmapTraining: public RmapTrainingBase
{
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

  /** \brief Load sample set from ROS bag. */
  void loadBag(const std::string& bag_path);

  /** \brief Save SVM model. */
  void loadSVM();

  /** \brief Train SVM. */
  void train();

  /** \brief Predict SVM. */
  void predict();

  /** \brief Publish marker array. */
  void publishMarkerArray() const;

 protected:
  //! Sample list
  std::vector<SampleVector> sample_list_;

  //! Whether SVM model is loaded from file
  bool svm_loaded_;
  //! path of SVM model file
  std::string svm_path_;

  //! SVM input node which is used for prediction
  svm_node input_node_[input_dim_ + 1];
  //! SVM input node list which is used for training
  svm_node* all_input_nodes_;
  //! SVM problem
  svm_problem svm_prob_;
  //! SVM parameter
  svm_parameter svm_param_;
  //! SVM model
  svm_model *svm_mo_;

  //! Support vector coefficients
  Eigen::VectorXd svm_coeff_vec_;
  //! Support vector matrix
  Eigen::Matrix<double, input_dim_, Eigen::Dynamic> svm_sv_mat_;

  //! Whether to use libsvm function for prediction
  static constexpr bool use_libsvm_prediction_ = false;

  //! Grid map
  std::shared_ptr<grid_map::GridMap> grid_map_;

  //! ROS related members
  ros::NodeHandle nh_;

  ros::Publisher rmap_cloud_pub_;
  ros::Publisher marker_arr_pub_;
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

namespace mc_rtc
{
/*! \brief Configuration loader. */
template<>
struct ConfigurationLoader<DiffRmap::RmapTrainingBase::Configuration>
{
  static DiffRmap::RmapTrainingBase::Configuration load(
      const mc_rtc::Configuration & mc_rtc_config)
  {
    DiffRmap::RmapTrainingBase::Configuration inst_config;
    mc_rtc_config("grid_map_margin_ratio", inst_config.grid_map_margin_ratio);
    mc_rtc_config("grid_map_resolution", inst_config.grid_map_resolution);
    mc_rtc_config("grid_map_height_scale", inst_config.grid_map_height_scale);
    mc_rtc_config("xy_plane_height", inst_config.xy_plane_height);
    return inst_config;
  }
};
}
