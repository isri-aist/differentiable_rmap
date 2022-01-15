/* Author: Masaki Murooka */

#pragma once

#include <mc_rtc/Configuration.h>

#include <ros/ros.h>
#include <differentiable_rmap/RmapGridSet.h>

#include <libsvm/svm.h>

#include <differentiable_rmap/SamplingUtils.h>
#include <differentiable_rmap/RosUtils.h>


namespace DiffRmap
{
/** \brief Virtual base class to plan in sample space based on differentiable reachability map. */
class RmapVisualizationBase
{
 public:
  /** \brief Configure from mc_rtc configuration.
      \param mc_rtc_config mc_rtc configuration
   */
  virtual void configure(const mc_rtc::Configuration& mc_rtc_config) = 0;

  /** \brief Setup visualization.
      \param grid_bag_path path of ROS bag file of grid set (file for output if load_grid is false, input otherwise)
      \param load_grid whether to load grid set from file
   */
  virtual void setup(const std::string& grid_bag_path = "/tmp/rmap_grid_set.bag",
                     bool load_grid = false) = 0;

  /** \brief Run visualization once. */
  virtual void runOnce() = 0;

  /** \brief Setup and run visualization loop.
      \param grid_bag_path path of ROS bag file of grid set (file for output if load_grid is false, input otherwise)
      \param load_grid whether to load grid set from file
   */
  virtual void runLoop(const std::string& grid_bag_path = "/tmp/rmap_grid_set.bag",
                       bool load_grid = false) = 0;
};

/** \brief Class to plan in sample space based on differentiable reachability map.
    \tparam SamplingSpaceType sampling space
*/
template <SamplingSpace SamplingSpaceType>
class RmapVisualization: public RmapVisualizationBase
{
 public:
  /*! \brief Configuration. */
  struct Configuration
  {
    //! Threshold of SVM predict value to be determined as reachable
    double svm_thre = 0.0;

    //! Grid resolution for position [m]
    double pos_resolution = 0.1;

    //! Grid resolution for rotation [rad]
    double rot_resolution = 0.1;

    /*! \brief Load mc_rtc configuration. */
    inline void load(const mc_rtc::Configuration& mc_rtc_config)
    {
      mc_rtc_config("svm_thre", svm_thre);
      mc_rtc_config("pos_resolution", pos_resolution);
      mc_rtc_config("rot_resolution", rot_resolution);
    }
  };

 public:
  /*! \brief Dimension of sample. */
  static constexpr int sample_dim_ = sampleDim<SamplingSpaceType>();

  /*! \brief Dimension of SVM input. */
  static constexpr int input_dim_ = inputDim<SamplingSpaceType>();

 public:
  /*! \brief Type of sample vector. */
  using SampleType = Sample<SamplingSpaceType>;

  /*! \brief Type of input vector. */
  using InputType = Input<SamplingSpaceType>;

 public:
  /** \brief Constructor.
      \param sample_bag_path path of ROS bag file of sample set
      \param svm_path path of SVM model file
   */
  RmapVisualization(const std::string& sample_bag_path = "/tmp/rmap_sample_set.bag",
                    const std::string& svm_path = "/tmp/rmap_svm_model.libsvm");

  /** \brief Destructor. */
  ~RmapVisualization();

  /** \brief Configure from mc_rtc configuration.
      \param mc_rtc_config mc_rtc configuration
   */
  virtual void configure(const mc_rtc::Configuration& mc_rtc_config) override;

  /** \brief Setup visualization.
      \param grid_bag_path path of ROS bag file of grid set (file for output if load_grid is false, input otherwise)
      \param load_grid whether to load grid set from file
   */
  virtual void setup(const std::string& grid_bag_path = "/tmp/rmap_grid_set.bag",
                     bool load_grid = false) override;

  /** \brief Run visualization once. */
  virtual void runOnce() override;

  /** \brief Setup and run visualization loop.
      \param grid_bag_path path of ROS bag file of grid set (file for output if load_grid is false, input otherwise)
      \param load_grid whether to load grid set from file
   */
  virtual void runLoop(const std::string& grid_bag_path = "/tmp/rmap_grid_set.bag",
                       bool load_grid = false) override;

 protected:
  /** \brief Load sample set from ROS bag. */
  void loadSampleSet(const std::string& sample_bag_path);

  /** \brief Save SVM model. */
  void loadSVM(const std::string& svm_path);

  /** \brief Load grid set. */
  void loadGridSet(const std::string& grid_bag_path);

  /** \brief Dump generated grid set to ROS bag. */
  void dumpGridSet(const std::string& grid_bag_path);

  /** \brief Publish marker array. */
  void publishMarkerArray() const;

 protected:
  //! mc_rtc Configuration
  mc_rtc::Configuration mc_rtc_config_;

  //! Configuration
  Configuration config_;

  //! Min/max position of samples
  SampleType sample_min_;
  SampleType sample_max_;

  //! SVM model
  svm_model *svm_mo_;

  //! Support vector coefficients
  Eigen::VectorXd svm_coeff_vec_;
  //! Support vector matrix
  Eigen::Matrix<double, input_dim_, Eigen::Dynamic> svm_sv_mat_;

  //! Grid set message
  differentiable_rmap::RmapGridSet grid_set_msg_;

  //! ROS related members
  ros::NodeHandle nh_;

  ros::Publisher marker_arr_pub_;
};

/** \brief Create RmapVisualization instance.
    \param sampling_space sampling space
    \param sample_bag_path path of ROS bag file of sample set
    \param svm_path path of SVM model file
*/
std::shared_ptr<RmapVisualizationBase> createRmapVisualization(
    SamplingSpace sampling_space,
    const std::string& sample_bag_path = "/tmp/rmap_sample_set.bag",
    const std::string& svm_path = "/tmp/rmap_svm_model.libsvm");
}
