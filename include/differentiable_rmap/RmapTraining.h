/* Author: Masaki Murooka */

#pragma once

#include <mc_rtc/Configuration.h>

#include <ros/ros.h>
#include <std_msgs/Float64.h>
#include <grid_map_ros/grid_map_ros.hpp>
#include <grid_map_msgs/GridMap.h>

#include <libsvm/svm.h>

#include <differentiable_rmap/SamplingUtils.h>
#include <differentiable_rmap/RosUtils.h>


namespace DiffRmap
{
/** \brief Virtual base class to train SVM for differentiable reachability map. */
class RmapTrainingBase
{
 public:
  /** \brief Configure from mc_rtc configuration.
      \param mc_rtc_config mc_rtc configuration
   */
  virtual void configure(const mc_rtc::Configuration& mc_rtc_config) = 0;

  /** \brief Setup SVM training. */
  virtual void setup() = 0;

  /** \brief Run SVM training once. */
  virtual void runOnce() = 0;

  /** \brief Setup and run SVM training loop. */
  virtual void runLoop() = 0;
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
    //! Margin ratio of grid map
    double grid_map_margin_ratio = 0.5;

    //! Resolution of grid map [m]
    double grid_map_resolution = 0.02;

    //! Height scale of grid map
    double grid_map_height_scale = 1.0;

    //! Theta threshold for slicing SE2 sample [deg]
    double slice_se2_theta_thre = 2.5;

    //! Z threshold for slicing R3 sample [m]
    double slice_r3_z_thre = 0.1;

    //! Z threshold for slicing SE3 sample [m]
    double slice_se3_z_thre = 0.1;

    //! Theta threshold for slicing SE3 sample [deg]
    double slice_se3_theta_thre = 20;

    /*! \brief Load mc_rtc configuration. */
    inline void load(const mc_rtc::Configuration& mc_rtc_config)
    {
      mc_rtc_config("grid_map_margin_ratio", grid_map_margin_ratio);
      mc_rtc_config("grid_map_resolution", grid_map_resolution);
      mc_rtc_config("grid_map_height_scale", grid_map_height_scale);
      mc_rtc_config("slice_se2_theta_thre", slice_se2_theta_thre);
      mc_rtc_config("slice_r3_z_thre", slice_r3_z_thre);
      mc_rtc_config("slice_se3_z_thre", slice_se3_z_thre);
      mc_rtc_config("slice_se3_theta_thre", slice_se3_theta_thre);
    }
  };

 public:
  /*! \brief Dimension of sample. */
  static constexpr int sample_dim_ = sampleDim<SamplingSpaceType>();

  /*! \brief Dimension of SVM input. */
  static constexpr int input_dim_ = inputDim<SamplingSpaceType>();

  //! Whether to use libsvm function for SVM prediction
  static constexpr bool use_libsvm_prediction_ = false;

 public:
  /*! \brief Type of sample vector. */
  using SampleType = Sample<SamplingSpaceType>;

  /*! \brief Type of input vector. */
  using InputType = Input<SamplingSpaceType>;

 public:
  /** \brief Constructor.
      \param bag_path path of ROS bag file
      \param svm_path path of SVM model file (file for output if load_svm is false, input otherwise)
      \param load_svm whether to load SVM model from file
   */
  RmapTraining(const std::string& bag_path = "/tmp/rmap_sample_set.bag",
               const std::string& svm_path = "/tmp/rmap_svm_model.libsvm",
               bool load_svm = false);

  /** \brief Destructor. */
  ~RmapTraining();

  /** \brief Configure from mc_rtc configuration.
      \param mc_rtc_config mc_rtc configuration
   */
  virtual void configure(const mc_rtc::Configuration& mc_rtc_config) override;

  /** \brief Setup SVM training. */
  virtual void setup() override;

  /** \brief Run SVM training once. */
  virtual void runOnce() override;

  /** \brief Setup and run SVM training loop. */
  virtual void runLoop() override;

  /** \brief Test SVM value calculation.
      \param[out] svm_value_libsvm SVM value calculated by libsvm
      \param[out] svm_value_eigen  SVM value calculated by Eigen
      \param[in] sample
   */
  void testCalcSVMValue(double& svm_value_libsvm,
                        double& svm_value_eigen,
                        const SampleType& sample) const;

  /** \brief Test SVM grad calculation.
      \param[out] svm_grad_analytical SVM grad calculated analytically
      \param[out] svm_grad_numerical  SVM grad calculated numerically
      \param[in] sample
   */
  void testCalcSVMGrad(Eigen::Ref<Vel<SamplingSpaceType>> svm_grad_analytical,
                       Eigen::Ref<Vel<SamplingSpaceType>> svm_grad_numerical,
                       const SampleType& sample) const;

 protected:
  /** \brief Setup SVM parameter. */
  void setupSVMParam();

  /** \brief Update SVM parameter. */
  void updateSVMParam();

  /** \brief Update origin of slicing. */
  void updateSliceOrigin();

  /** \brief Setup grid map. */
  void setupGridMap();

  /** \brief Load sample set from ROS bag. */
  void loadSampleSet(const std::string& bag_path);

  /** \brief Save SVM model. */
  void loadSVM();

  /** \brief Train SVM. */
  void trainSVM();

  /** \brief Predict SVM on grid map. */
  void predictOnSlicePlane();

  /** \brief Publish sliced cloud. */
  void publishSlicedCloud() const;

  /** \brief Publish marker array. */
  void publishMarkerArray() const;

 protected:
  //! mc_rtc Configuration
  mc_rtc::Configuration mc_rtc_config_;

  //! Configuration
  Configuration config_;

  //! Sample list
  std::vector<SampleType> sample_list_;

  //! Reachability list
  std::vector<bool> reachability_list_;

  //! Min/max position of samples
  SampleType sample_min_;
  SampleType sample_max_;

  //! Whether SVM model is loaded from file
  bool load_svm_;
  //! path of SVM model file
  std::string svm_path_;

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

  //! Grid map
  std::shared_ptr<grid_map::GridMap> grid_map_;

  //! Origin of slicing
  sva::PTransformd slice_origin_ = sva::PTransformd::Identity();

  //! Whether SVM training is required
  bool train_required_; // initial value depends on constructor argument

  //! Whether SVM training is updated
  bool train_updated_ = false;

  //! Whether origin of slicing is updated
  bool slice_updated_ = false;

  //! ROS related members
  ros::NodeHandle nh_;

  ros::Publisher reachable_cloud_pub_;
  ros::Publisher unreachable_cloud_pub_;
  ros::Publisher sliced_reachable_cloud_pub_;
  ros::Publisher sliced_unreachable_cloud_pub_;
  ros::Publisher marker_arr_pub_;
  ros::Publisher grid_map_pub_;

  std::shared_ptr<SubscVariableManager<std_msgs::Float64, double>> svm_thre_manager_;
  std::shared_ptr<SubscVariableManager<std_msgs::Float64, double>> svm_gamma_manager_;
  std::shared_ptr<SubscVariableManager<std_msgs::Float64, double>> svm_nu_manager_;
  std::shared_ptr<SubscVariableManager<std_msgs::Float64, double>> slice_z_manager_;
  std::shared_ptr<SubscVariableManager<std_msgs::Float64, double>> slice_roll_manager_;
  std::shared_ptr<SubscVariableManager<std_msgs::Float64, double>> slice_pitch_manager_;
  std::shared_ptr<SubscVariableManager<std_msgs::Float64, double>> slice_yaw_manager_;
};

/** \brief Create RmapTraining instance.
    \param sampling_space sampling space
    \param bag_path path of ROS bag file (empty for loading trained SVM model directly)
    \param svm_path path of SVM model file (file for output if load_svm is false, input otherwise)
    \param load_svm whether to load SVM model from file
*/
std::shared_ptr<RmapTrainingBase> createRmapTraining(
    SamplingSpace sampling_space,
    const std::string& bag_path = "/tmp/rmap_sample_set.bag",
    const std::string& svm_path = "/tmp/rmap_svm_model.libsvm",
    bool load_svm = false);
}
