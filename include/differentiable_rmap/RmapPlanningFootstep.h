/* Author: Masaki Murooka */

#pragma once

#include <sch/CD/CD_Pair.h>
#include <sch/S_Object/S_Box.h>

#include <differentiable_rmap/RmapPlanning.h>


namespace DiffRmap
{
/** \brief Class to plan footstep sequence based on differentiable reachability map.
    \tparam SamplingSpaceType sampling space
*/
template <SamplingSpace SamplingSpaceType>
class RmapPlanningFootstep: public RmapPlanning<SamplingSpaceType>
{
 public:
  /*! \brief Configuration of collision shape. */
  struct CollisionShapeConfiguration
  {
    //! Pose
    sva::PTransformd pose = sva::PTransformd::Identity();

    //! Scale
    Eigen::Vector3d scale = Eigen::Vector3d::Ones();

    /*! \brief Load mc_rtc configuration. */
    inline void load(const mc_rtc::Configuration& mc_rtc_config)
    {
      mc_rtc_config("pose", pose);
      mc_rtc_config("scale", scale);
    }
  };

  /*! \brief Configuration. */
  struct Configuration: public RmapPlanning<SamplingSpaceType>::Configuration
  {
    //! Number of footsteps
    int footstep_num = 3;

    //! Adjacent regularization weight
    double adjacent_reg_weight = 1e-3;

    //! Whether to switch left and right alternately (supported only in SE2)
    //! Suppose that the SVM model represents a reachable map from the right foot to the left foot.
    bool alternate_lr = true;

    //! Margin distance of collision avoidance [m]
    double collision_margin = 0.0;

    //! Foot shape configuration (used for collision avoidance with obstacles)
    CollisionShapeConfiguration foot_shape_config;

    //! List of obstacle shape configuration
    std::vector<CollisionShapeConfiguration> obst_shape_config_list = {};

    //! Vertices of foot marker
    std::vector<Eigen::Vector3d> foot_vertices = {
      Eigen::Vector3d(-0.1, -0.05, 0.0),
      Eigen::Vector3d(0.1, -0.05, 0.0),
      Eigen::Vector3d(0.1, 0.05, 0.0),
      Eigen::Vector3d(-0.1, 0.05, 0.0)
    };

    /*! \brief Load mc_rtc configuration. */
    inline void load(const mc_rtc::Configuration& mc_rtc_config)
    {
      RmapPlanning<SamplingSpaceType>::Configuration::load(mc_rtc_config);

      mc_rtc_config("footstep_num", footstep_num);
      mc_rtc_config("adjacent_reg_weight", adjacent_reg_weight);
      mc_rtc_config("alternate_lr", alternate_lr);
      mc_rtc_config("collision_margin", collision_margin);
      if (mc_rtc_config.has("foot_shape_config")) {
        foot_shape_config.load(mc_rtc_config("foot_shape_config"));
      }
      if (mc_rtc_config.has("obst_shape_config_list")) {
        for (const mc_rtc::Configuration& mc_rtc_obst_config : mc_rtc_config("obst_shape_config_list")) {
          CollisionShapeConfiguration obst_config;
          obst_config.load(mc_rtc_obst_config);
          obst_shape_config_list.push_back(obst_config);
        }
      }
      mc_rtc_config("foot_vertices", foot_vertices);
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
  RmapPlanningFootstep(const std::string& svm_path = "/tmp/rmap_svm_model.libsvm",
                       const std::string& bag_path = "/tmp/rmap_grid_set.bag");

  /** \brief Destructor. */
  ~RmapPlanningFootstep();

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

 protected:
  /** \brief Publish marker array. */
  virtual void publishMarkerArray() const override;

  /** \brief Publish current state. */
  virtual void publishCurrentState() const override;

  /** \brief Returns whether switching left and right foot alternately is supported. */
  static inline constexpr bool isAlternateSupported()
  {
    return SamplingSpaceType == SamplingSpace::SE2;
  }

 protected:
  //! Configuration
  Configuration config_;

  //! Current sample sequence
  std::vector<SampleType> current_sample_seq_;

  //! Sample corresponding to identity pose
  const SampleType identity_sample_ = poseToSample<SamplingSpaceType>(sva::PTransformd::Identity());

  //! Adjacent regularization matrix
  Eigen::MatrixXd adjacent_reg_mat_;

  //! Sch box of foot
  std::shared_ptr<sch::S_Box> foot_sch_;

  //! Sch box list of obstacles
  std::vector<std::shared_ptr<sch::S_Box>> obst_sch_list_;

  //! List of collision detector of sch objects
  std::vector<std::shared_ptr<sch::CD_Pair>> sch_cd_list_;

  //! List of closest points of sch objects
  std::vector<std::array<Eigen::Vector3d, 2>> closest_points_list_;

  //! Collision direction from obstacle to foot
  Eigen::Vector3d collision_dir_ = Eigen::Vector3d::Zero();

  //! ROS related members
  ros::Publisher current_pose_arr_pub_;
  ros::Publisher current_poly_arr_pub_;
  // Separate topics to change the marker colors on the right and left feet
  ros::Publisher current_left_poly_arr_pub_;
  ros::Publisher current_right_poly_arr_pub_;

 protected:
  // See https://stackoverflow.com/a/6592617
  using RmapPlanning<SamplingSpaceType>::mc_rtc_config_;

  using RmapPlanning<SamplingSpaceType>::sample_min_;
  using RmapPlanning<SamplingSpaceType>::sample_max_;

  using RmapPlanning<SamplingSpaceType>::svm_mo_;

  using RmapPlanning<SamplingSpaceType>::qp_coeff_;
  using RmapPlanning<SamplingSpaceType>::qp_solver_;

  using RmapPlanning<SamplingSpaceType>::target_sample_;

  using RmapPlanning<SamplingSpaceType>::svm_coeff_vec_;
  using RmapPlanning<SamplingSpaceType>::svm_sv_mat_;

  using RmapPlanning<SamplingSpaceType>::grid_set_msg_;

  using RmapPlanning<SamplingSpaceType>::nh_;

  using RmapPlanning<SamplingSpaceType>::marker_arr_pub_;
};

/** \brief Create RmapPlanningFootstep instance.
    \param sampling_space sampling space
    \param svm_path path of SVM model file
    \param bag_path path of ROS bag file of grid set (empty for no grid set)
*/
std::shared_ptr<RmapPlanningBase> createRmapPlanningFootstep(
    SamplingSpace sampling_space,
    const std::string& svm_path = "/tmp/rmap_svm_model.libsvm",
    const std::string& bag_path = "/tmp/rmap_grid_set.bag");
}
