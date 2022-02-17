/* Author: Masaki Murooka */

#pragma once

#include <map>
#include <unordered_map>

#include <mc_rtc/constants.h>

#include <differentiable_rmap/RmapPlanning.h>
#include <differentiable_rmap/RobotUtils.h>

namespace DiffRmap
{
/** \brief Class to plan loco-manipulation motion based on differentiable reachability map.

    This class does not inherit RmapPlanning because it has many differences from RmapPlanning (e.g., it holds multiple
   SVM models).
 */
class RmapPlanningLocomanip
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

    //! Initial sample pose list
    std::map<Limb, sva::PTransformd> initial_sample_pose_list = {{Limb::LeftFoot, sva::PTransformd::Identity()},
                                                                 {Limb::RightFoot, sva::PTransformd::Identity()},
                                                                 {Limb::LeftHand, sva::PTransformd::Identity()},
                                                                 {Limb::RightHand, sva::PTransformd::Identity()}};

    //! Number of footsteps
    int motion_len = 3;

    //! Regularization weight
    double reg_weight = 1e-6;

    //! Adjacent regularization weight
    double adjacent_reg_weight = 1e-3;

    //! QP objective weight for SVM inequality error
    double svm_ineq_weight = 1e6;

    //! Center of hand arc trajectory [m]
    Eigen::Vector2d hand_traj_center = Eigen::Vector2d::Zero();

    //! Radius of hand arc trajectory [m]
    double hand_traj_radius = 1.0;

    //! Target angles of start and goal in hand arc trajectory [rad] ([deg] in YAML file)
    std::pair<double, double> target_hand_traj_angles = {0, -10};

    //! Height of hand markers [m]
    double hand_marker_height = 0.0;

    //! Vertices of foot marker
    std::vector<Eigen::Vector3d> foot_vertices = {Eigen::Vector3d(-0.1, -0.05, 0.0), Eigen::Vector3d(0.1, -0.05, 0.0),
                                                  Eigen::Vector3d(0.1, 0.05, 0.0), Eigen::Vector3d(-0.1, 0.05, 0.0)};

    /*! \brief Load mc_rtc configuration. */
    inline void load(const mc_rtc::Configuration & mc_rtc_config)
    {
      mc_rtc_config("loop_rate", loop_rate);
      mc_rtc_config("publish_interval", publish_interval);
      mc_rtc_config("svm_thre", svm_thre);
      mc_rtc_config("delta_config_limit", delta_config_limit);
      if(mc_rtc_config.has("initial_sample_pose_list"))
      {
        std::map<std::string, sva::PTransformd> tmp_initial_sample_pose_list;
        mc_rtc_config("initial_sample_pose_list", tmp_initial_sample_pose_list);
        for(const auto & tmp_initial_sample_pose_kv : tmp_initial_sample_pose_list)
        {
          initial_sample_pose_list[strToLimb(tmp_initial_sample_pose_kv.first)] = tmp_initial_sample_pose_kv.second;
        }
      }
      mc_rtc_config("motion_len", motion_len);
      mc_rtc_config("reg_weight", reg_weight);
      mc_rtc_config("adjacent_reg_weight", adjacent_reg_weight);
      mc_rtc_config("svm_ineq_weight", svm_ineq_weight);
      mc_rtc_config("hand_traj_center", hand_traj_center);
      mc_rtc_config("hand_traj_radius", hand_traj_radius);
      if(mc_rtc_config.has("target_hand_traj_angles"))
      {
        mc_rtc_config("target_hand_traj_angles", target_hand_traj_angles);
        target_hand_traj_angles.first = mc_rtc::constants::toRad(target_hand_traj_angles.first);
        target_hand_traj_angles.second = mc_rtc::constants::toRad(target_hand_traj_angles.second);
      }
      mc_rtc_config("hand_marker_height", hand_marker_height);
      mc_rtc_config("foot_vertices", foot_vertices);
    }
  };

public:
  /*! \brief Sampling space. */
  static constexpr SamplingSpace SamplingSpaceType = SamplingSpace::SE2;

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
      \param svm_path_list path list of SVM model file
      \param bag_path_list path list of ROS bag file of grid set (empty for no grid set)
   */
  RmapPlanningLocomanip(const std::unordered_map<Limb, std::string> & svm_path_list,
                        const std::unordered_map<Limb, std::string> & bag_path_list);

  /** \brief Destructor. */
  ~RmapPlanningLocomanip();

  /** \brief Configure from mc_rtc configuration.
      \param mc_rtc_config mc_rtc configuration
   */
  void configure(const mc_rtc::Configuration & mc_rtc_config);

  /** \brief Setup planning. */
  void setup();

  /** \brief Run planning once.
      \param publish whether to publish message
   */
  void runOnce(bool publish);

  /** \brief Setup and run planning loop. */
  void runLoop();

protected:
  /** \brief Get rmap planning for specified limb. */
  inline std::shared_ptr<RmapPlanning<SamplingSpaceType>> rmapPlanning(Limb limb) const
  {
    return std::dynamic_pointer_cast<RmapPlanning<SamplingSpaceType>>(rmap_planning_list_.at(limb));
  }

  /** \brief Calculate sample from hand trajectory. */
  SampleType calcSampleFromHandTraj(double angle) const;

  /** \brief Calculate sample gradient from hand trajectory. */
  SampleType calcSampleGradFromHandTraj(double angle) const;

  /** \brief Publish marker array. */
  void publishMarkerArray() const;

  /** \brief Publish current state. */
  void publishCurrentState() const;

  /** \brief Transform topic callback. */
  void transCallback(const geometry_msgs::TransformStamped::ConstPtr & trans_st_msg);

protected:
  //! Sample corresponding to identity pose
  static inline const Sample<SamplingSpaceType> identity_sample_ =
      poseToSample<SamplingSpaceType>(sva::PTransformd::Identity());

protected:
  //! mc_rtc Configuration
  mc_rtc::Configuration mc_rtc_config_;

  //! Configuration
  Configuration config_;

  //! List of rmap planning for each limb
  std::unordered_map<Limb, std::shared_ptr<RmapPlanningBase>> rmap_planning_list_;

  //! QP coefficients
  OmgCore::QpCoeff qp_coeff_;

  //! QP solver
  std::shared_ptr<OmgCore::QpSolver> qp_solver_;

  //! Current sample sequence for foot
  std::vector<Sample<SamplingSpaceType>> current_foot_sample_seq_;

  //! Current angle sequence in hand trajectory
  std::vector<double> current_hand_traj_angle_seq_;

  //! Current sample sequence for hand
  std::vector<Sample<SamplingSpaceType>> current_hand_sample_seq_;

  //! Start sample
  std::unordered_map<Limb, Sample<SamplingSpaceType>> start_sample_list_;

  //! Adjacent regularization matrix
  Eigen::MatrixXd adjacent_reg_mat_;

  //! Dimensions of configuration, SVM inequality, and collision inequality
  int config_dim_ = 0;
  int svm_ineq_dim_ = 0;
  int collision_ineq_dim_ = 0;

  //! Index of configuration where hand starts
  int hand_start_config_idx_ = 0;

  //! ROS related members
  ros::NodeHandle nh_;

  ros::Subscriber trans_sub_;
  ros::Publisher marker_arr_pub_;
  ros::Publisher current_pose_arr_pub_;
  ros::Publisher current_poly_arr_pub_;
  // Separate topics to change the marker colors on the right and left feet
  ros::Publisher current_left_poly_arr_pub_;
  ros::Publisher current_right_poly_arr_pub_;
  // Use cloud message to visualize sphere at hand position
  ros::Publisher current_cloud_pub_;
};
} // namespace DiffRmap
