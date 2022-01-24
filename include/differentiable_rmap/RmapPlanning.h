/* Author: Masaki Murooka */

#pragma once

#include <mc_rtc/Configuration.h>

#include <ros/ros.h>
#include <geometry_msgs/TransformStamped.h>
#include <grid_map_ros/grid_map_ros.hpp>
#include <grid_map_msgs/GridMap.h>
#include <differentiable_rmap/RmapGridSet.h>

#include <libsvm/svm.h>

#include <optmotiongen/Utils/QpUtils.h>

#include <differentiable_rmap/SamplingUtils.h>
#include <differentiable_rmap/RosUtils.h>


namespace DiffRmap
{
/** \brief Virtual base class to plan in sample space based on differentiable reachability map. */
class RmapPlanningBase
{
 public:
  /** \brief Configure from mc_rtc configuration.
      \param mc_rtc_config mc_rtc configuration
   */
  virtual void configure(const mc_rtc::Configuration& mc_rtc_config) = 0;

  /** \brief Setup planning. */
  virtual void setup() = 0;

  /** \brief Run planning once.
      \param publish whether to publish message
   */
  virtual void runOnce(bool publish) = 0;

  /** \brief Setup and run planning loop. */
  virtual void runLoop() = 0;
};

/** \brief Class to plan in sample space based on differentiable reachability map.
    \tparam SamplingSpaceType sampling space
*/
template <SamplingSpace SamplingSpaceType>
class RmapPlanning: public RmapPlanningBase
{
 public:
  /*! \brief Configuration. */
  struct Configuration
  {
    //! Rate in runLoop()
    int loop_rate = 2000;

    //! Step interval to publish in runLoop()
    int publish_interval = 20;

    //! Threshold of SVM predict value to be determined as reachable
    double svm_thre = 0.0;

    //! Limit of configuration update in one step [m], [rad]
    double delta_config_limit = 0.1;

    //! Initial sample pose
    sva::PTransformd initial_sample_pose = sva::PTransformd::Identity();

    //! Whether to predict on grid map
    bool grid_map_prediction = true;

    //! Margin ratio of grid map
    double grid_map_margin_ratio = 0.0;

    //! Resolution of grid map [m]
    double grid_map_resolution = 0.02;

    //! Height scale of grid map
    double grid_map_height_scale = 1.0;

    /*! \brief Load mc_rtc configuration. */
    inline void load(const mc_rtc::Configuration& mc_rtc_config)
    {
      mc_rtc_config("loop_rate", loop_rate);
      mc_rtc_config("publish_interval", publish_interval);
      mc_rtc_config("svm_thre", svm_thre);
      mc_rtc_config("delta_config_limit", delta_config_limit);
      mc_rtc_config("initial_sample_pose", initial_sample_pose);
      mc_rtc_config("grid_map_prediction", grid_map_prediction);
      mc_rtc_config("grid_map_margin_ratio", grid_map_margin_ratio);
      mc_rtc_config("grid_map_resolution", grid_map_resolution);
      mc_rtc_config("grid_map_height_scale", grid_map_height_scale);
    }
  };

 public:
  /*! \brief Dimension of sample. */
  static constexpr int sample_dim_ = sampleDim<SamplingSpaceType>();

  /*! \brief Dimension of SVM input. */
  static constexpr int input_dim_ = inputDim<SamplingSpaceType>();

  /*! \brief Dimension of velocity. */
  static constexpr int vel_dim_ = velDim<SamplingSpaceType>();

 public:
  /*! \brief Type of sample vector. */
  using SampleType = Sample<SamplingSpaceType>;

  /*! \brief Type of input vector. */
  using InputType = Input<SamplingSpaceType>;

  /*! \brief Type of velocity vector. */
  using VelType = Vel<SamplingSpaceType>;

 public:
  /** \brief Constructor.
      \param svm_path path of SVM model file
      \param bag_path path of ROS bag file of grid set (empty for no grid set)
  */
  RmapPlanning(const std::string& svm_path = "/tmp/rmap_svm_model.libsvm",
               const std::string& bag_path = "/tmp/rmap_grid_set.bag");

  /** \brief Constructor.
      \param svm_path path of SVM model file
      \param bag_path path of ROS bag file of grid set (empty for no grid set)
      \param setup_ros whether to setup ROS
   */
  RmapPlanning(const std::string& svm_path,
               const std::string& bag_path,
               bool setup_ros);

  /** \brief Destructor. */
  ~RmapPlanning();

  /** \brief Configure from mc_rtc configuration.
      \param mc_rtc_config mc_rtc configuration
   */
  virtual void configure(const mc_rtc::Configuration& mc_rtc_config) override;

  /** \brief Setup planning. */
  virtual void setup() override;

  /** \brief Run planning once.
      \param publish whether to publish message
   */
  virtual void runOnce(bool publish) override;

  /** \brief Setup and run planning loop. */
  virtual void runLoop() override;

  /** \brief Calculate SVM value.
      \param sample sample
  */
  double calcSVMValue(const SampleType& sample) const;

  /** \brief Calculate gradient of SVM value.
      \param sample sample
  */
  VelType calcSVMGrad(const SampleType& sample) const;

 protected:
  /** \brief Setup grid map. */
  void setupGridMap();

  /** \brief Load SVM model. */
  void loadSVM(const std::string& svm_path);

  /** \brief Load grid set. */
  void loadGridSet(const std::string& bag_path);

  /** \brief Predict SVM on grid map. */
  void predictOnSlicePlane();

  /** \brief Publish marker array. */
  virtual void publishMarkerArray() const;

  /** \brief Publish current state. */
  virtual void publishCurrentState() const;

  /** \brief Transform topic callback. */
  virtual void transCallback(const geometry_msgs::TransformStamped::ConstPtr& trans_st_msg);

 public:
  //! Min/max position of samples
  SampleType sample_min_ = SampleType::Constant(-1.0);
  SampleType sample_max_ = SampleType::Constant(1.0);

  //! SVM model
  svm_model* svm_mo_;

  //! Support vector coefficients
  Eigen::VectorXd svm_coeff_vec_;

  //! Support vector matrix
  Eigen::Matrix<double, input_dim_, Eigen::Dynamic> svm_sv_mat_;

  //! Grid set message
  differentiable_rmap::RmapGridSet::ConstPtr grid_set_msg_;

 protected:
  //! mc_rtc Configuration
  mc_rtc::Configuration mc_rtc_config_;

  //! Configuration
  Configuration config_;

  //! QP coefficients
  OmgCore::QpCoeff qp_coeff_;

  //! QP solver
  std::shared_ptr<OmgCore::QpSolver> qp_solver_;

  //! Current sample
  SampleType current_sample_ = poseToSample<SamplingSpaceType>(sva::PTransformd::Identity());

  //! Target sample
  SampleType target_sample_ = poseToSample<SamplingSpaceType>(sva::PTransformd::Identity());

  //! Grid map
  std::shared_ptr<grid_map::GridMap> grid_map_;

  //! ROS related members
  ros::NodeHandle nh_;

  ros::Subscriber trans_sub_;
  ros::Publisher marker_arr_pub_;
  ros::Publisher grid_map_pub_;
  ros::Publisher current_pos_pub_;
  ros::Publisher current_pose_pub_;
};

/** \brief Create RmapPlanning instance.
    \param sampling_space sampling space
    \param svm_path path of SVM model file
    \param bag_path path of ROS bag file of grid set (empty for no grid set)
*/
std::shared_ptr<RmapPlanningBase> createRmapPlanning(
    SamplingSpace sampling_space,
    const std::string& svm_path = "/tmp/rmap_svm_model.libsvm",
    const std::string& bag_path = "/tmp/rmap_grid_set.bag");
}
